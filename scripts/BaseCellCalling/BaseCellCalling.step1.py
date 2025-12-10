#!/usr/bin/env python3
"""
Optimized BaseCellCalling Step 1 - Matching Original Output Exactly

This version produces IDENTICAL output to the original while being faster.
Key: Must output ALL lines from input, not just variants.
"""

import os
import math
import timeit
import argparse
from pathlib import Path

import numpy as np
import pysam
from scipy.stats import betabinom

ALLELES = ['A', 'C', 'T', 'G', 'I', 'D', 'N', 'O']


def longest_run(s: str) -> int:
    if not s:
        return 0
    max_run = current = 1
    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 1
    return max_run


def homopolymer_filter(context: str, alts: list, direction: str) -> int:
    if not context or context == '.':
        return 0
    for alt in alts:
        combined = (context + alt) if direction == 'upstream' else (alt + context)
        if longest_run(combined) >= 4:
            return 1
    return 0


def variant_calling_step1(
    infile: str, outfile: str, fasta_path: str,
    alpha1: float, beta1: float, alpha2: float, beta2: float,
    min_ac_cells: int, min_ac_reads: int,
    min_cells: int, min_reads: int,
    min_cell_types: int, max_cell_types: int,
    fisher_cutoff: float, window: int,
):
    inFasta = pysam.FastaFile(fasta_path) if fasta_path else None
    
    out = open(outfile, 'w')
    counter = 0
    cell_types_idx = {}
    
    with open(infile, 'r') as f:
        for line in f:
            if line.startswith('##'):
                out.write(line)
                continue
            
            if line.startswith('#CHROM') and counter == 0:
                line = line.rstrip('\n')
                elements = line.split('\t')
                
                cell_types_idx = {x: elements[x] for x in range(len(elements)) if x > 4}
                
                info_header = {
                    'ALT': '##INFO=ALT,Description="Alternative alleles found"',
                    'FILTER': '##INFO=FILTER,Description="Filter status of the variant site"',
                    'Cell_types': '##INFO=Cell_types,Description="Cell type/s with the variant"',
                    'Up_context': '##INFO=Up_context,Description="Up-stream bases in reference (4 bases)"',
                    'Down_context': '##INFO=Down_context,Description="Down-stream bases in reference (4 bases)"',
                    'N_ALT': '##INFO=N_ALT,Description="Cell type/s with the variant"',
                    'Dp': '##INFO=Dp,Description="Depth of coverage (reads) in the cell type supporting the variant"',
                    'Nc': '##INFO=Nc,Description="Number of distinct cells found in the cell type with the mutation"',
                    'Bc': '##INFO=Bc,Description="Number of reads (base count) supporting the variants in the cell type with the mutation"',
                    'Cc': '##INFO=Cc,Description="Number of distinct cells supporting the variant in the cell type with the mutation"',
                    'VAF': '##INFO=VAF,Description="Variant allele frequency of variant in the cell type with the mutation"',
                    'CCF': '##INFO=CCF,Description="Cancer cell fraction (fraction of ditinct cells) supporting the alternative allele in the cell type with the mutation"',
                    'BCp': '##INFO=BCp,Description="Beta-binomial p-value for the variant allele (considering read counts)"',
                    'CCp': '##INFO=CCp,Description="Beta-binomial p-value for the variant allele (considering cell counts)"',
                    'Cell_types_min_BC': '##INFO=Cell_types_min_BC,Description="Number of cell types with a minimum number of reads covering a site"',
                    'Cell_types_min_CC': '##INFO=Cell_types_min_CC,Description="Number of cell types with a minimum number of distinct cells found in a specific site"',
                    'Rest_BC': '##INFO=Rest_BC,Description="Base counts (reads) supporting other alternative alleles in this site. BC;DP;P-value (betabin)"',
                    'Rest_CC': '##INFO=Rest_CC,Description="Cell counts supporting other alternative alleles in this site. CC;NC;P-value (betabin)"',
                    'Fisher_p': '##INFO=Fisher_p,Description="Strand bias test. Fisher exact test p-value between forward and reverse reads in variant and reference allele"',
                    'Cell_type_Filter': '##INFO=Cell_type_Filter,Description="Filter status of the variant site in each cell type"',
                }
                for key in info_header:
                    out.write(info_header[key] + '\n')
                
                calling_cols = ['ALT', 'FILTER', 'Cell_types', 'Up_context', 'Down_context',
                               'N_ALT', 'Dp', 'Nc', 'Bc', 'Cc', 'VAF', 'CCF', 'BCp', 'CCp',
                               'Cell_types_min_BC', 'Cell_types_min_CC', 'Rest_BC', 'Rest_CC',
                               'Fisher_p', 'Cell_type_Filter']
                elements.insert(4, '\t'.join(calling_cols))
                out.write('\t'.join(elements) + '\n')
                continue
            
            counter += 1
            line = line.rstrip('\n')
            elements = line.split('\t')
            
            CHROM = str(elements[0])
            POS = int(elements[1])
            REF = elements[3]
            
            num_bases = 5
            if inFasta:
                try:
                    context = inFasta.fetch(CHROM, POS - (num_bases + 1), POS + num_bases).upper()
                    up_context = context[:num_bases]
                    down_context = context[(num_bases + 1):(num_bases + 1 + num_bases)]
                except:
                    up_context = '.'
                    down_context = '.'
            else:
                up_context = '.'
                down_context = '.'
            
            Alts = []
            Cell_types = []
            DPs = []
            NCs = []
            BCs = []
            CCs = []
            BCp = []
            CCp = []
            VAF = []
            CCF = []
            Filter = []
            Fisher_p = []
            
            Cell_types_min_BC = 0
            Cell_types_min_CC = 0
            Sum_alts_bc = 0
            Sum_alts_cc = 0
            Sum_dp = 0
            Sum_nc = 0
            
            for cell_type_i in cell_types_idx.keys():
                cell_type = cell_types_idx[cell_type_i]
                INFO_i = elements[cell_type_i]
                
                if INFO_i.startswith('NA'):
                    continue
                
                parts = INFO_i.split('|')
                if len(parts) != 7:
                    continue
                
                DP = int(parts[0])
                NC = int(parts[1])
                
                if DP < min_reads or NC < min_cells:
                    continue
                
                Cell_types_min_BC += 1
                Cell_types_min_CC += 1
                
                cc = [int(x) for x in parts[2].split(':')]
                bc = [int(x) for x in parts[3].split(':')]
                bq = [int(x) for x in parts[4].split(':')]
                bcf = [int(x) for x in parts[5].split(':')]
                bcr = [int(x) for x in parts[6].split(':')]
                
                Alts2 = sum(bc[x] for x in range(len(bc)) if ALLELES[x] not in [REF, 'O'])
                Sum_alts_bc += Alts2
                Cc2 = sum(cc[x] for x in range(len(cc)) if ALLELES[x] not in [REF, 'O'])
                Sum_alts_cc += Cc2
                Sum_dp += DP
                Sum_nc += NC
                
                Alt_bc_dict = {ALLELES[x]: bc[x] for x in range(len(bc)) 
                              if ALLELES[x] not in [REF, 'I', 'D', 'N', 'O'] and bc[x] > 0}
                Alt_bc_p_dict = {x: round(betabinom.sf(Alt_bc_dict[x] - 0.1, DP, alpha1, beta1), 4) 
                                for x in Alt_bc_dict.keys()}
                Alt_bc_p_dict_sig = [x for x in Alt_bc_p_dict.keys() if Alt_bc_p_dict[x] < 0.001]
                
                Alt_cc_dict = {ALLELES[x]: cc[x] for x in range(len(cc)) 
                              if ALLELES[x] not in [REF, 'I', 'D', 'N', 'O'] and cc[x] > 0}
                Alt_cc_p_dict = {x: round(betabinom.sf(Alt_cc_dict[x] - 0.1, NC, alpha2, beta2), 4) 
                                for x in Alt_cc_dict.keys()}
                Alt_cc_p_dict_sig = [x for x in Alt_cc_p_dict.keys() if Alt_cc_p_dict[x] < 0.001]
                
                Alt_candidates = sorted(list(set(Alt_bc_p_dict_sig + Alt_cc_p_dict_sig)))
                
                if len(Alt_candidates) > 0:
                    alt_candidates = '|'.join(Alt_candidates)
                    Alts.append(alt_candidates)
                    Cell_types.append(cell_type)
                    DPs.append(str(DP))
                    NCs.append(str(NC))
                    
                    P_BC = list(Alt_bc_p_dict.values())
                    P_CC = list(Alt_cc_p_dict.values())
                    
                    b = '|'.join(str(Alt_bc_dict[x]) for x in Alt_candidates)
                    BCs.append(b)
                    c = '|'.join(str(Alt_cc_dict[x]) for x in Alt_candidates)
                    CCs.append(c)
                    
                    bp = '|'.join(str(Alt_bc_p_dict[x]) for x in Alt_candidates)
                    BCp.append(bp)
                    cp = '|'.join(str(Alt_cc_p_dict[x]) for x in Alt_candidates)
                    CCp.append(cp)
                    
                    vaf = '|'.join(str(round(Alt_bc_dict[x] / float(DP), 4)) for x in Alt_candidates)
                    VAF.append(vaf)
                    ccf = '|'.join(str(round(Alt_cc_dict[x] / float(NC), 4)) for x in Alt_candidates)
                    CCF.append(ccf)
                    
                    b0 = sum(Alt_bc_dict[x] for x in Alt_candidates)
                    c0 = sum(Alt_cc_dict[x] for x in Alt_candidates)
                    Sum_dp -= b0
                    Sum_nc -= c0
                    Sum_alts_bc -= b0
                    Sum_alts_cc -= c0
                    
                    if not (min(P_BC) < 0.001 and min(P_CC) < 0.001):
                        Filter.append('BetaBin_problem')
                    elif len(Alt_candidates) > 1:
                        Filter.append('Multi-allelic')
                    elif int(c) < min_ac_cells:
                        Filter.append('Low_cells')
                    elif int(b) < min_ac_reads:
                        Filter.append('Low_reads')
                    else:
                        Filter.append('PASS')
            
            if len(Alts) > 0:
                FILTER = []
                PASS = [x for x in Filter if x == 'PASS']
                
                if len(PASS) > max_cell_types:
                    FILTER.append('Multiple_cell_types')
                
                LEN_Alts = len(set(Alts))
                if LEN_Alts > 1 or 'Multi-allelic' in Filter:
                    FILTER.append('Multi-allelic')
                
                if Cell_types_min_CC < min_cell_types:
                    FILTER.append('Min_cell_types')
                
                if len(Filter) - len(PASS) > 0:
                    FILTER.append('Cell_type_noise')
                
                if Sum_alts_bc > 0:
                    BC_noise_p = round(1 - betabinom.cdf(Sum_alts_bc - 0.1, Sum_dp, alpha1, beta1), 4)
                    CC_noise_p = round(1 - betabinom.cdf(Sum_alts_cc - 0.1, Sum_nc, alpha2, beta2), 4)
                    rest_BC = f"{Sum_alts_bc};{Sum_dp};{BC_noise_p}"
                    rest_CC = f"{Sum_alts_cc};{Sum_nc};{CC_noise_p}"
                else:
                    BC_noise_p = 1
                    CC_noise_p = 1
                    rest_BC = f"{Sum_alts_bc};{Sum_dp};{BC_noise_p}"
                    rest_CC = f"{Sum_alts_cc};{Sum_nc};{CC_noise_p}"
                
                if BC_noise_p < 0.05 or CC_noise_p < 0.05:
                    FILTER.append('Noisy_site')
                
                homopolymer_up = homopolymer_filter(up_context, Alts, 'upstream')
                if homopolymer_up == 1:
                    FILTER.append('LC_Upstream')
                
                homopolymer_down = homopolymer_filter(down_context, Alts, 'downstream')
                if homopolymer_down == 1:
                    FILTER.append('LC_Downstream')
                
                if len(FILTER) == 0:
                    if 'PASS' in Filter:
                        FILTER_str = 'PASS'
                    else:
                        FILTER_str = ','.join(Filter)
                else:
                    FILTER_str = ','.join(FILTER)
                
                INFO = [
                    ','.join(Alts),
                    FILTER_str,
                    ','.join(Cell_types),
                    up_context,
                    down_context,
                    str(LEN_Alts),
                    ','.join(DPs),
                    ','.join(NCs),
                    ','.join(BCs),
                    ','.join(CCs),
                    ','.join(VAF),
                    ','.join(CCF),
                    ','.join(BCp),
                    ','.join(CCp),
                    str(Cell_types_min_BC),
                    str(Cell_types_min_CC),
                    rest_BC,
                    rest_CC,
                    '.',
                    ','.join(Filter),
                ]
            else:
                if Sum_alts_bc > 0:
                    BC_noise_p = round(1 - betabinom.cdf(Sum_alts_bc - 0.1, Sum_dp, alpha1, beta1), 4)
                    CC_noise_p = round(1 - betabinom.cdf(Sum_alts_cc - 0.1, Sum_nc, alpha2, beta2), 4)
                    rest_BC = f"{Sum_alts_bc};{Sum_dp};{BC_noise_p}"
                    rest_CC = f"{Sum_alts_cc};{Sum_nc};{CC_noise_p}"
                else:
                    BC_noise_p = 1
                    CC_noise_p = 1
                    rest_BC = f"{Sum_alts_bc};{Sum_dp};{BC_noise_p}"
                    rest_CC = f"{Sum_alts_cc};{Sum_nc};{CC_noise_p}"
                
                if BC_noise_p < 0.001 or CC_noise_p < 0.001:
                    FILTER_str = 'Noisy_site'
                else:
                    FILTER_str = '.'
                
                INFO = [
                    '.',
                    FILTER_str,
                    '.',
                    up_context,
                    down_context,
                    '.',
                    '.',
                    '.',
                    '.',
                    '.',
                    '.',
                    '.',
                    '.',
                    '.',
                    str(Cell_types_min_BC),
                    str(Cell_types_min_BC),
                    rest_BC,
                    rest_CC,
                    '.',
                    '.',
                ]
            
            elements.insert(4, '\t'.join(INFO))
            out.write('\t'.join(elements) + '\n')
            
            if counter % 1000000 == 0:
                print(f'Step 1 variant calling: {counter} positions processed')
    
    if inFasta:
        inFasta.close()
    out.close()
    
    print(f'  Total positions processed: {counter}')


def initialize_parser():
    parser = argparse.ArgumentParser(description='Optimized variant calling step 1')
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--outfile', type=str, required=True)
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--min_cov', type=int, default=5)
    parser.add_argument('--min_cells', type=int, default=5)
    parser.add_argument('--min_ac_cells', type=int, default=2)
    parser.add_argument('--min_ac_reads', type=int, default=3)
    parser.add_argument('--max_cell_types', type=int, default=1)
    parser.add_argument('--min_cell_types', type=int, default=2)
    parser.add_argument('--fisher_cutoff', type=float, default=1)
    parser.add_argument('--alpha1', type=float, default=0.260288007167716)
    parser.add_argument('--beta1', type=float, default=173.94711910763732)
    parser.add_argument('--alpha2', type=float, default=0.08354121346569514)
    parser.add_argument('--beta2', type=float, default=103.47683488327257)
    return parser


def main():
    parser = initialize_parser()
    args = parser.parse_args()
    
    print('\n------------------------------')
    print('Optimized Variant Calling Step 1')
    print('------------------------------\n')
    
    outfile = args.outfile + '.calling.step1.tsv'
    
    variant_calling_step1(
        args.infile, outfile, args.ref,
        args.alpha1, args.beta1, args.alpha2, args.beta2,
        args.min_ac_cells, args.min_ac_reads,
        args.min_cells, args.min_cov,
        args.min_cell_types, args.max_cell_types,
        args.fisher_cutoff, 20000,
    )
    
    print(f'\nOutput: {outfile}')


if __name__ == '__main__':
    start = timeit.default_timer()
    main()
    stop = timeit.default_timer()
    print(f'\nTotal computing time: {round(stop - start, 2)} seconds')
