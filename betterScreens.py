#!/usr/local/src/miniconda/bin/python

import pandas as pd
import numpy as np
import re
import sys, getopt
from Bio import SeqIO
from Bio.SeqIO.FastaIO import SimpleFastaParser
from Bio.SeqUtils.ProtParam import ProteinAnalysis

FandP = {"A" : 0.31, "C" : 1.54, "D" : -0.77, "E" : -0.64, "F" : 1.79, "G" : 0, "H" : 0.13, "I" : 1.8, "K" : -0.99,
           "L" : 1.7, "M" : 1.23, "N" : -0.6, "P" : 0.72, "Q" : -0.22, "R" : -1.01, "S" : -0.04, "T" : 0.26, "V" : 1.22,
                    "W" : 2.25, "Y" : 0.96}

## length
def aaLength(seq):
    return [len(i) for i in seq]

## Amino Acid Composition => Number of each amino acid
def aaComp(seq):
    def prota(s):
        X = ProteinAnalysis(s)
        return X.get_amino_acids_percent()
    return[prota(i) for i in seq]
    
## N-term_amino_acid => What the N-term amino acid is
def ntermAA(seq):
    return[i[0] for i in seq]

## Charge => The charge (D,E and K,R)
def chargeAA(seq):
    def prota(s):
        X = ProteinAnalysis(s)
        c = X.count_amino_acids()
        return c['K'] + c['R'] - c['D'] - c['E']
    return[prota(i) for i in seq]

## Charge_Density => Charge / Length
def chDenstity(ch,l):
    return(ch/l)

## Average positive charge spacing
##    1. Identify the position of all positively charged residues in the sequence
##       a. if <2 charged residues then NA
##    2. Starting at the first charged residue, identify the intervening sequence(s)
##    3. Work out the length of each intervening sequence
##    4. Calculate an average
def chSpacing(seq):
    def avspace(s):
        spaces = [len(i) for i in re.findall(r'[KR](\w*?)[KR]', s)]
        return "NA" if len(spaces) == 0 else np.average(spaces)        
    return[avspace(i) for i in seq]

## Consecutive_R
## Consecutive K
## Consecutive charged => Consecutive positive charge
##    1. Identify all runs of R of length 1-n
##    2. Report max(n)
def consecAA(seq,aa):
    def consec(s,aa):
        c = [len(i) for i in re.findall(r'({}+)'.format(aa), s)]
        return "NA" if len(c) == 0 else np.max(c)
    return[consec(i,aa) for i in seq]

## Hydrophobicity (F&P) => Hydrophobicity using the Faucher and Pliska scales
def hydroFandP(seq,FandP):
    def protfp(s,FandP):
        #total = 0
        #[total := total + FandP[i] for i in s]
        sumt = []
        [sumt.append(FandP[i]) for i in s]
        total = np.sum(sumt)
        hfp = total / len(s)
        return(hfp)
    return[protfp(i,FandP) for i in seq]

## %hydrophobic_residues => Percentage of hydrophobic residues (A, C, F, G, H, I, L, M, P, T, V, W, Y)
def hydroPerc(seq):
    def proth(s):
        X = ProteinAnalysis(s)
        c = X.count_amino_acids()
        return (c['A'] + c['C'] + c['F'] + c['G'] + c['H'] + c['I'] + c['L'] + c['M'] + c['P'] + c['T'] + c['V'] + c['W'] + c['Y']) / len(s)
    return[proth(i) for i in seq]

## Amphipathicity_max => Maximum hmoment
## Amphipathicity_avg => Average hmoment

def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print('betterScreens.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('betterScreens.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg

    getProps(inputfile,outputfile)

def getProps(inputfile,outputfile):

    df = pd.DataFrame(columns=['seqID','abund','desc','sequence'])
    with open(inputfile,"r") as fasta_file:
        for titledesc, sequence in SimpleFastaParser(fasta_file):
            title,abund,desc = titledesc.split(None)
            df = df.append({
                "seqID": title,
                "abund": abund,
                "desc": desc,
                "sequence": sequence
                }, ignore_index=True)
            
    # returns the length of the sequence
    df['length'] = aaLength(df['sequence'])
    # this function returns a list of dictionaries
    # I can't find a way to assign this as additional columns to the existing df
    # so I am making a new df and concatenating it to the other
    # the order should be intact bby index
    df2 = pd.DataFrame(aaComp(df['sequence']))
    df = pd.concat([df, df2], axis=1, sort=False)
    # function returns the first letter of the sequence
    df['ntermAA'] = ntermAA(df['sequence'])
    # function returns the charge based only on D,E K,R
    df['chargeAA'] = chargeAA(df['sequence'])
    # function returns the charge density ch/len
    df['chDenstity'] = chDenstity(df['chargeAA'],df['length'])
    # function returns the average spacing of positively charged residues
    df['chSpacing'] = chSpacing(df['sequence'])
    # function returns maximum run of n
    # K
    df['consK'] = consecAA(df['sequence'],'K')
    # R
    df['consR'] = consecAA(df['sequence'],'R')
    # [KR]
    df['consKR'] = consecAA(df['sequence'],'[KR]')
    # function returns fauchere and pliska average hydrophobicity
    df['hydroFandP'] = hydroFandP(df['sequence'],FandP)
    # function returns 
    df['hydroPerc'] = hydroPerc(df['sequence'])

    df.to_csv(outputfile, index=False)

if __name__ == "__main__":
   main(sys.argv[1:])