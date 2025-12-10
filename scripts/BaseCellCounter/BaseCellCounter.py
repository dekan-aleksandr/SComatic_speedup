#!/usr/bin/env python3
"""
Optimized BaseCellCounter - 2025 High-Performance Version

Key optimizations over original:
1. Use pysam bulk operations (get_query_sequences, get_query_qualities)
2. Faster barcode lookup with set-based operations
3. Avoid redundant list comprehensions
4. Use truncate=True to avoid processing outside the region
5. Streamlined counting logic
"""

import pysam
import timeit
import pybedtools
import multiprocessing as mp
import argparse
import glob
import os
import time


def EasyReadPileup(pileup_list, ref_base):
    """Optimized pileup parsing."""
    bases = {'A', 'C', 'T', 'G', 'N'}
    new_list = []
    ac = 0
    ref_upper = ref_base.upper()
    
    for x in pileup_list:
        upper = x.upper()
        if upper in bases:
            new_list.append(upper)
            if upper != ref_upper:
                ac += 1
        elif len(x) > 1:
            if x[1] == '-':
                new_list.append('D')
                ac += 1
            elif x[1] == '+':
                new_list.append('I')
                ac += 1
            else:
                new_list.append('NA')
        elif x == '*':
            new_list.append('O')
        else:
            new_list.append('NA')
    
    return new_list, ac


def run_interval_optimized(interval, BAM, FASTA, MIN_COV, MIN_CC, MIN_AF, MIN_AC, tmp_dir, BQ, MQ, MAX_DP):
    """Optimized interval processing matching original logic but faster."""
    
    CHROM = interval[0]
    START = int(interval[1])
    END = int(interval[2])
    
    bam = pysam.AlignmentFile(BAM, "rb")
    inFasta = pysam.FastaFile(FASTA)
    
    POSITIONS = []
    
    pileup_iter = bam.pileup(
        CHROM, START, END,
        min_base_quality=BQ,
        min_mapping_quality=MQ,
        max_depth=MAX_DP if MAX_DP > 0 else 1000000,
        ignore_overlaps=False,
        truncate=True,
    )
    
    for p in pileup_iter:
        POS = p.pos
        
        ref_base = inFasta.fetch(CHROM, POS, POS + 1).upper()
        if ref_base == 'N':
            continue
        
        DP = p.get_num_aligned()
        if DP < MIN_COV:
            continue
        
        PILEUP_LIST = p.get_query_sequences(mark_matches=True, add_indels=True)
        NEW_PILEUP_LIST, AC = EasyReadPileup(PILEUP_LIST, ref_base)
        
        CALLABLE = [x for x in NEW_PILEUP_LIST if x != 'NA']
        if len(CALLABLE) < MIN_COV or AC < MIN_AC:
            continue
        
        QUALITIES = p.get_query_qualities()
        reads = p.pileups
        
        BASE_COUNTS = {'A': 0, 'C': 0, 'T': 0, 'G': 0, 'D': 0, 'I': 0, 'N': 0, 'O': 0}
        BASE_QUALITIES = {'A': 0, 'C': 0, 'T': 0, 'G': 0, 'D': 0, 'I': 0, 'N': 0, 'O': 0}
        CELL_COUNTS = {'A': [], 'C': [], 'T': [], 'G': [], 'D': [], 'I': [], 'N': [], 'O': []}
        BASE_COUNTS_f = {'A': 0, 'C': 0, 'T': 0, 'G': 0, 'D': 0, 'I': 0, 'N': 0, 'O': 0}
        BASE_COUNTS_r = {'A': 0, 'C': 0, 'T': 0, 'G': 0, 'D': 0, 'I': 0, 'N': 0, 'O': 0}
        
        CELLS = []
        count = 0
        n_reads = len(reads)
        
        for read_i in range(n_reads):
            read_alignment = reads[read_i].alignment
            
            try:
                barcode = read_alignment.get_tag("CB").split("-")[0]
            except (KeyError, AttributeError):
                continue
            
            if read_alignment.is_secondary or read_alignment.is_duplicate or read_alignment.is_supplementary:
                continue
            
            base = NEW_PILEUP_LIST[read_i]
            if base not in BASE_COUNTS:
                continue
            
            bq = QUALITIES[read_i]
            count += 1
            
            BASE_COUNTS[base] += 1
            BASE_QUALITIES[base] += bq
            
            if read_alignment.is_reverse:
                BASE_COUNTS_r[base] += 1
            else:
                BASE_COUNTS_f[base] += 1
            
            CELL_COUNTS[base].append(barcode)
            CELLS.append(barcode)
        
        if count < MIN_COV:
            continue
        
        NC = len(set(CELLS))
        if NC < MIN_CC:
            continue
        
        CELL_COUNTS2 = {x: len(set(CELL_COUNTS[x])) for x in CELL_COUNTS}
        
        alleles = ['A', 'C', 'T', 'G', 'I', 'D']
        BC_str = ':'.join(str(BASE_COUNTS[a]) for a in alleles)
        BQ_str = ':'.join(str(BASE_QUALITIES[a]) for a in alleles)
        BCf_str = ':'.join(str(BASE_COUNTS_f[a]) for a in alleles)
        BCr_str = ':'.join(str(BASE_COUNTS_r[a]) for a in alleles)
        CC_str = ':'.join(str(CELL_COUNTS2[a]) for a in alleles)
        
        info_field = '|'.join(['DP', 'NC', 'CC', 'BC', 'BQ', 'BCf', 'BCr'])
        INFO = '|'.join([str(count), str(NC), CC_str, BC_str, BQ_str, BCf_str, BCr_str])
        
        LINE = '\t'.join([str(CHROM), str(POS + 1), ref_base, info_field, INFO])
        POSITIONS.append(LINE)
    
    inFasta.close()
    bam.close()
    
    ID = '__'.join([str(CHROM), str(START), str(END)])
    out_temp = os.path.join(tmp_dir, ID + '.BaseCellCounts.temp')
    return [out_temp, '\n'.join(POSITIONS)]


def collect_result(result):
    if result[1] != '':
        with open(result[0], 'w') as out:
            out.write(result[1])


def concatenate_sort_temp_files_and_write(out_file, tmp_dir, ID):
    all_files = glob.glob(os.path.join(tmp_dir, '*.BaseCellCounts.temp'))
    
    if not all_files:
        print('No temporary files found')
        return
    
    file_dict = {}
    for filename in all_files:
        basename = os.path.basename(filename)
        coordinates = basename.replace('.BaseCellCounts.temp', '')
        CHROM, START, END = coordinates.split('__')
        START = int(START)
        
        if CHROM not in file_dict:
            file_dict[CHROM] = {}
        file_dict[CHROM][START] = filename
    
    date = time.strftime("%d/%m/%Y")
    header = ['#CHROM', 'POS', 'REF', 'INFO', str(ID)]
    concepts = """##INFO=DP,Description="Depth of coverage">
##INFO=NC,Description="Number of different cells">
##INFO=CC,Description="Cell counts [A:C:T:G:I:D:N:O]">
##INFO=BC,Description="Base counts [A:C:T:G:I:D:N:O]">
##INFO=BQ,Description="Base quality sums [A:C:T:G:I:D:N:O]">
##INFO=BCf,Description="Base counts in forward reads [A:C:T:G:I:D:N:O]">
##INFO=BCr,Description="Base counts in reverse reads [A:C:T:G:I:D:N:O]">"""
    
    with open(out_file, 'w') as out:
        out.write(f"##fileDate={date}\n")
        out.write(concepts + '\n')
        out.write('\t'.join(header) + '\n')
        
        for chrom in sorted(file_dict.keys()):
            for start in sorted(file_dict[chrom].keys()):
                filename = file_dict[chrom][start]
                with open(filename, 'r') as f:
                    content = f.read().strip()
                    if content:
                        out.write(content + '\n')
                os.remove(filename)


def MakeWindows(CONTIG, FASTA, bed, bed_out, window):
    if bed == '':
        inFasta = pysam.FastaFile(FASTA)
        CONTIG_Names = inFasta.references
        LIST = [(x, 1, inFasta.get_reference_length(x)) for x in CONTIG_Names]
        a = pybedtools.BedTool(LIST)
        inFasta.close()
    else:
        a = pybedtools.BedTool(bed)
    
    if CONTIG != 'all':
        a2 = a.filter(lambda b: b.chrom == CONTIG)
    else:
        a2 = a
    
    if bed_out != '':
        b = pybedtools.BedTool(bed_out)
        a3 = a2.subtract(b)
    else:
        a3 = a2
    
    return a3.window_maker(a3, w=window)


def initialize_parser():
    parser = argparse.ArgumentParser(description='Optimized base/cell counter')
    parser.add_argument('--bam', type=str, required=True)
    parser.add_argument('--ref', type=str, required=True)
    parser.add_argument('--chrom', type=str, required=True)
    parser.add_argument('--out_folder', default='.')
    parser.add_argument('--id')
    parser.add_argument('--nprocs', default=1, type=int)
    parser.add_argument('--bin', type=int, default=50000)
    parser.add_argument('--bed', type=str, default='')
    parser.add_argument('--bed_out', type=str, default='')
    parser.add_argument('--min_ac', type=int, default=0)
    parser.add_argument('--min_af', type=float, default=0)
    parser.add_argument('--min_dp', type=int, default=5)
    parser.add_argument('--min_cc', type=int, default=5)
    parser.add_argument('--min_bq', type=int, default=20)
    parser.add_argument('--min_mq', type=int, default=255)
    parser.add_argument('--max_dp', type=int, default=8000)
    parser.add_argument('--tmp_dir', type=str, default='.')
    return parser


def main():
    parser = initialize_parser()
    args = parser.parse_args()
    
    ID = args.id or os.path.basename(args.bam).replace('.bam', '')
    out_file = os.path.join(args.out_folder, f"{ID}.tsv")
    print(f"Outfile: {out_file}\n")
    
    if args.tmp_dir != '.':
        os.makedirs(args.tmp_dir, exist_ok=True)
    
    BED = MakeWindows(args.chrom, args.ref, args.bed, args.bed_out, args.bin)
    
    if args.nprocs > 1:
        pool = mp.Pool(args.nprocs)
        for row in BED:
            pool.apply_async(
                run_interval_optimized,
                args=(row, args.bam, args.ref, args.min_dp, args.min_cc, 
                      args.min_af, args.min_ac, args.tmp_dir, args.min_bq, 
                      args.min_mq, args.max_dp),
                callback=collect_result
            )
        pool.close()
        pool.join()
    else:
        for row in BED:
            result = run_interval_optimized(
                row, args.bam, args.ref, args.min_dp, args.min_cc,
                args.min_af, args.min_ac, args.tmp_dir, args.min_bq,
                args.min_mq, args.max_dp
            )
            collect_result(result)
    
    concatenate_sort_temp_files_and_write(out_file, args.tmp_dir, ID)


if __name__ == '__main__':
    start = timeit.default_timer()
    main()
    stop = timeit.default_timer()
    print(f'Computation time: {round(stop - start)} seconds')
