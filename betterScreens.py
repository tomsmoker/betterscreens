#!/usr/local/src/miniconda/bin/python

import pandas as pd
import numpy as np
import re
import collections
import sys, getopt
#from Bio import SeqIO
#from Bio.SeqIO.FastaIO import SimpleFastaParser
#from Bio.SeqUtils.ProtParam import ProteinAnalysis

FandP = {"A" : 0.31, "C" : 1.54, "D" : -0.77, "E" : -0.64, "F" : 1.79, "G" : 0, "H" : 0.13, "I" : 1.8, "K" : -0.99,
           "L" : 1.7, "M" : 1.23, "N" : -0.6, "P" : 0.72, "Q" : -0.22, "R" : -1.01, "S" : -0.04, "T" : 0.26, "V" : 1.22,
                    "W" : 2.25, "Y" : 0.96}
Eisen = {"A" : 0.62, "R" : -2.53, "N" : -0.78, "D" : -0.90, "C" : 0.29, "Q" : -0.85, "E" : -0.74, "G" : 0.48, "H" : -0.40,
           "I" : 1.38, "L" : 1.06, "K" : -1.50, "M" : 0.64, "F" : 1.19, "P" : 0.12, "S" : -0.18, "T" : -0.05, "W" : 0.81,
                    "Y" : 0.26, "V" : 1.08}

## length
def aaLength(seq):
    return [len(i) for i in seq]

## Amino Acid Composition => Number of each amino acid
#def aaComp(seq):
#    def prota(s):
#        X = ProteinAnalysis(s)
#        return X.get_amino_acids_percent()
#    return[prota(i) for i in seq]

## alternative using collections.Counter
def aaCompCount(seq):
     def prota(s):
        c = collections.Counter(s)
        seqd = {letter:c[letter]/len(s) for letter in 'ARNDCQEGHILKMFPSTWYV'}
        return seqd
     return[prota(i) for i in seq]

## N-term_amino_acid => What the N-term amino acid is
def ntermAA(seq):
    return[i[0] for i in seq]

## Charge => The charge (D,E and K,R)
#def chargeAA(seq):
#    def prota(s):
#        X = ProteinAnalysis(s)
#        c = X.count_amino_acids()
#        return c['K'] + c['R'] - c['D'] - c['E']
#    return[prota(i) for i in seq]

## alternative using collections.Counter
def chargeAACount(seq):
    def prota(s):
       c = collections.Counter(s)
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
        [sumt.append(FandP.get(i,0)) for i in s]
        total = np.sum(sumt)
        hfp = total / len(s)
        return(hfp)
    return[protfp(i,FandP) for i in seq]

## %hydrophobic_residues => Percentage of hydrophobic residues (A, C, F, G, H, I, L, M, P, T, V, W, Y)
#def hydroPerc(seq):
#    def proth(s):
#        X = ProteinAnalysis(s)
#        c = X.count_amino_acids()
#        return (c['A'] + c['C'] + c['F'] + c['G'] + c['H'] + c['I'] + c['L'] + c['M'] + c['P'] + c['T'] + c['V'] + c['W'] + c['Y']) / len(s)
#    return[proth(i) for i in seq]

## alternative using collections.Counter
def hydroPercCount(seq):
    def prota(s):
        c = collections.Counter(s)
        return (c['A'] + c['C'] + c['F'] + c['G'] + c['H'] + c['I'] + c['L'] + c['M'] + c['P'] + c['T'] + c['V'] + c['W'] + c['Y']) / len(s)
    return[prota(i) for i in seq]

## Amphipathicity_max => Maximum hmoment
## Amphipathicity_avg => Average hmoment
def hmoment(seq):
    def runhmoment(seq, angle=100, window=11, scale=Eisen):
        if len(seq) < window:
            return ("NA","NA")
        
        def seq2scale(seq,scale):
            return [scale.get(i,0) for i in seq]

        pep = [seq2scale(seq[i:i+window][::-1], scale) for i in range(len(seq)-(window-1))]
        pep = np.array(pep)

        #angle = [angle * (np.pi / 180) * i for i in range(1,min(len(pep)+1,window+1))]
        angle = [angle * (np.pi / 180) * i for i in range(1,window+1)]
        vcos = pep * np.cos(angle)
        vsin = pep * np.sin(angle)
        vcos = vcos.sum(axis=1)
        vsin = vsin.sum(axis=1)
        #hm = np.sqrt(vsin * vsin + vcos * vcos) / min(len(pep),window)
        hm = np.sqrt(vsin * vsin + vcos * vcos) / window

        return max(hm), np.mean(hm)
    return[runhmoment(i) for i in seq]


## main
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

    df = pd.read_csv(inputfile,sep=" ",names=['seqID','abund','desc','sequence'])
#    df = pd.DataFrame(columns=['seqID','abund','desc','sequence'])
#    with open(inputfile,"r") as fasta_file:
#        for titledesc, sequence in SimpleFastaParser(fasta_file):
#            title,abund,desc = titledesc.split(None)
#            df = df.append({
#                "seqID": title,
#                "abund": abund,
#                "desc": desc,
#                "sequence": sequence
#                }, ignore_index=True)
    
    print("stored sequences")

    # returns the length of the sequence
    df['length'] = aaLength(df['sequence'])
    print("computed lengths")
    # this function returns a list of dictionaries
    # I can't find a way to assign this as additional columns to the existing df
    # so I am making a new df and concatenating it to the other
    # the order should be intact bby index
    #df2 = pd.DataFrame(aaComp(df['sequence']))
    df2 = pd.DataFrame(aaCompCount(df['sequence']))
    df = pd.concat([df, df2], axis=1, sort=False)
    print("computed composition")
    # function returns the first letter of the sequence
    df['ntermAA'] = ntermAA(df['sequence'])
    print("computed ntermAA")
    # function returns the charge based only on D,E K,R
    #df['chargeAA'] = chargeAA(df['sequence'])
    df['chargeAA'] = chargeAACount(df['sequence'])
    print("computed chargeAA")
    # function returns the charge density ch/len
    df['chDenstity'] = chDenstity(df['chargeAA'],df['length'])
    print("computed chDensity")
    # function returns the average spacing of positively charged residues
    df['chSpacing'] = chSpacing(df['sequence'])
    print("computed chSpacing")
    # function returns maximum run of n
    # K
    df['consK'] = consecAA(df['sequence'],'K')
    # R
    df['consR'] = consecAA(df['sequence'],'R')
    # [KR]
    df['consKR'] = consecAA(df['sequence'],'[KR]')
    print("computed consecutiveAA")
    # function returns fauchere and pliska average hydrophobicity
    df['hydroFandP'] = hydroFandP(df['sequence'],FandP)
    print("computed FandP")
    # function returns 
    #df['hydroPerc'] = hydroPerc(df['sequence'])
    df['hydroPerc'] = hydroPercCount(df['sequence'])
    print("computed hydroPerc")
    # returns the average and mean amphipathicity
    df['amph_max'], df["amph_avg"] = zip(*hmoment(df['sequence']))
    print("computed amphipathicity")

    df.to_csv(outputfile, index=False)

if __name__ == "__main__":
   main(sys.argv[1:])
