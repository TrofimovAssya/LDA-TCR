import pickle
import seaborn as sns
import pandas as pd
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation as LDA



def build_parser():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--filter1', default='', help='Selection of TCRs to include in the kmer counts')
    parser.add_argument('--filterlower', default=[1,5,10], type=int, nargs='+', help='nb kmer abundance per person to act as filter')
    parser.add_argument('--filterupper', default=[0,2], type=int, nargs='+', help='nb kmer abundance per person to act as filter')
    parser.add_argument('--nb_lda_components', default=[1,2,3,4,5,6,7,8,9,10],
                        type=int, nargs='+', help='nb LDA components')
    parser.add_argument('--vjgene', default=False, help='V and J genes instead of kmers')
    parser.add_argument('--k', default=3,type=int, help='size of kmers')
    parser.add_argument('--minlength', default=7,type=int, help='minimal length of tcr')
    parser.add_argument('--nowildcard', default=True, help='remove wildcard and stop codons')
    parser.add_argument('--seq_type', default='aa',help='type of sequence')
    parser.add_argument('--cached_dataset',default='', help='dataset')
    parser.add_argument('--dataset_name',default='', help='dataset')
    return parser


def get_all_kmers(seqlist,k):
    kmer_list = []
    for seq in tqdm(seqlist):
        seq = seq.strip()
        for i in range(0,len(seq)-k+1):
            kmer_list.append(seq[i:i+k])
    kmer_list = list(set(kmer_list))
    return kmer_list

def pt_abundance(cdr3s,kmers,abundance,k):
    kmer_dict = {}
    ###initializing kmer dict
    for kmer in kmers:
        kmer_dict[kmer] = 0
    ### populating the dictionnary
    for seq,qty in zip(cdr3s,abundance):
        for i in range(0,len(seq)-k+1):
            kmer = seq[i:i+k]
            if kmer in kmer_dict:
                kmer_dict[kmer]+=qty
    ### moving dictionary data to kmer vector
    counts = []
    for kmer in kmers:
         counts.append(kmer_dict[kmer])
    return counts

def pt_abundance_vj(cdr3s_v, cdr3s_j,kmers,abundance):
    kmer_dict = {}
    ###initializing kmer dict
    for kmer in kmers:
        kmer_dict[kmer] = 0
    ### populating the dictionnary
    #for cdr3s in [cdr3s_v, cdr3s_j]:
    for cdr3s in [ cdr3s_j]:
        for kmer,qty in zip(cdr3s,abundance):
            kmer_dict[kmer]+=qty
    ### moving dictionary data to kmer vector
    counts = []
    for kmer in kmers:
         counts.append(kmer_dict[kmer])
    return counts


def load_patient(pt_name,fdir,selected_tcrs=None, seq_type = 'aa'):
    ### this function loads a set of TCRs either for NT or AA kmers
    a = pd.read_csv(f'{fdir}{pt_name}',sep='\t')
    if not selected_tcrs==None:
        if seq_type=='aa':
            a = a[a['cdr3aa'].isin(selected_tcrs)]
        else:
            a = a[a['cdr3nt'].isin(selected_tcrs)]
    return a


def load_patient_vj(pt_name,fdir,selected_tcrs=None):
    ### this function loads a set of TCRs for VDJ lda
    a = pd.read_csv(f'{fdir}{pt_name}',sep='\t')
    if not selected_tcrs==None:
        a = a[['v','j']]
    return a

def main(argv=None):

    opt = build_parser().parse_args(argv)

    print ('Getting the TCR set...')

    pheno_britanova = pd.read_csv('../DATA/metadata.txt',sep='\t')

    if not opt.vjgene:

        if opt.filter1=='':

            all_tcr_aa = []
            fdir = '../DATA/'
            if opt.seq_type=='nt':
                all_tcr_nt = []


            for fname in tqdm(pheno_britanova['file_name']):
                a = pd.read_csv(f'{fdir}{fname}',sep='\t')
                all_tcr_aa+=list(a['cdr3aa'])
                if opt.seq_type=='nt':
                    all_tcr_nt+=list(a['cdr3nt'])


            ### removing TCRs under a certain length
            minlen = opt.minlength
            keep = [len(i)>minlen for i in pd.Series(list(all_tcr_aa))]
            if opt.seq_type=='nt':
                all_tcr_nt = list(pd.Series(list(all_tcr_nt))[keep])
            all_tcr_aa = list(pd.Series(list(all_tcr_aa))[keep])


            ### removing TCRs with wildcard or stop 
            if opt.nowildcard:
                keep = [not '*' in i and not  '_' in i  for i in pd.Series(list(all_tcr_aa))]
                all_tcr_aa = list(pd.Series(list(all_tcr_aa))[keep])
                if opt.seq_type=='nt':
                    all_tcr_nt = list(pd.Series(list(all_tcr_nt))[keep])

            if opt.seq_type=='aa':
                all_tcr = set(all_tcr_aa)
            else:
                all_tcr = set(all_tcr_nt)
            selected_tcrs = None

        else:
            all_tcr = pd.read_csv(opt.filter1, index_col=0)
            all_tcr = all_tcr['0']
            selected_tcrs = list(all_tcr)

        print ('Getting all the kmers....')
        k=opt.k
        kmer_list = get_all_kmers(all_tcr,k)
        nb_kmers = len(kmer_list)


        print (f'Total number of kmers: {nb_kmers}')

    elif opt.vjgene:

        all_tcr_v = []
        all_tcr_j = []
        fdir = '../DATA/'

        # filtering using the same criteria as the aminoacid ones
        for fname in tqdm(pheno_britanova['file_name']):
            a = pd.read_csv(f'{fdir}{fname}',sep='\t')
            all_tcr_v+=list(a['v'])
            all_tcr_j+=list(a['j'])

        kmer_list = list(set(all_tcr_v+all_tcr_j))
        kmer_list = list(set(all_tcr_j))
        nb_kmers = len(kmer_list)

        print (f'Total number of V and J genes: {nb_kmers}')

    else:
        print ('Not implemented')

    print ('Getting the kmers for the whole Britanova cohort')
    if not opt.vjgene:
        if opt.cached_dataset=='' :

            print ('No cached data...')
            print ('Parsing data and assembling dataset...')
            test_abundance = pd.DataFrame([])
            if opt.seq_type=='aa':
                column = 'cdr3aa'
            else:
                column = 'cdr3nt'

            for fname in tqdm(pheno_britanova['file_name']):
                a = load_patient(fname,fdir,selected_tcrs,opt.seq_type)
                if test_abundance.empty:
                    test_abundance = pd.DataFrame(pt_abundance(a[column],kmer_list,a['freq'],k))
                    test_abundance.index = kmer_list
                    test_abundance.columns = [fname]
                else:
                    test_abundance[fname] = pt_abundance(a[column],kmer_list,a['freq'],k)

            test_abundance = test_abundance.T
            test_abundance.to_csv(f'cached_data/{opt.dataset_name}')
        else:
            test_abundance = pd.read_csv(f'./cached_data/{opt.cached_dataset}',index_col=0)

    elif opt.vjgene:
        if opt.cached_dataset=='':
            print ('No cached data...')
            print ('Parsing data and assembling dataset...')
            test_abundance = pd.DataFrame([])
            for fname in tqdm(pheno_britanova['file_name']):
                #fname = fname.split('.txt')[0]
                a = load_patient_vj(fname,fdir)
                if test_abundance.empty:
                    test_abundance = pd.DataFrame(pt_abundance_vj(a['v'],a['j'],kmer_list,a['freq']))
                    test_abundance.index = kmer_list
                    test_abundance.columns = [fname]
                else:
                    test_abundance[fname] = pt_abundance_vj(a['v'],a['j'],kmer_list,a['freq'])

            test_abundance = test_abundance.T
            test_abundance.to_csv(f'cached_data/{opt.dataset_name}')
        else:
            print ('Found cached data...')
            test_abundance = pd.read_csv(f'./cached_data/{opt.cached_dataset}',index_col=0)



    overzero = np.sum(test_abundance>0,axis=0)/test_abundance.shape[0]
    assert np.max(overzero)==1

    for filt_l in opt.filterlower:
        for filt_u in opt.filterupper:
            print (f'Filter = {filt} %')

            save_dir = f'{opt.seq_type}LDA_{k}_results_filter_{filt_l}L{filt_u}U'
            os.mkdir(save_dir)

            print ('filtering kmers')
            if filt_l>0:
                lower = filt_l/79
                upper = (79-filt_u)/79

                overzero = np.sum(test_abundance>0,axis=0)/test_abundance.shape[0]
                keep_columns = overzero.index[np.logical_and(overzero<upper, overzero>lower)]
                #keep_columns = overzero.index[overzero>lower]
                print (f'From {test_abundance.shape[1]} kmers')
                test_abundance = test_abundance[keep_columns]
                print (f'Keeping {test_abundance.shape[1]} kmers')
            test_abundance.to_csv(f'{save_dir}/{opt.seq_type}LDA_{k}_filteredkmer_abundance.csv')


            for nb_lda_components in opt.nb_lda_components:
                print (f'LDA with {nb_lda_components} components')
                lda = LDA(n_components=nb_lda_components,verbose = 1)
                lda.fit(test_abundance)

                probabilities = lda.transform(test_abundance)
                probabilities = pd.DataFrame(probabilities)
                probabilities['ptnames'] = pheno_britanova['file_name']

                print ('Saving probabilities...')
                probabilities.to_csv(f'{save_dir}/{opt.seq_type}LDA_{k}_britanova_lda{nb_lda_components}.csv')

                print ('Saving model...')
                filename = f'{save_dir}/{opt.seq_type}LDA_{k}_finalized_model_lda{nb_lda_components}.mdl'
                pickle.dump(lda, open(filename, 'wb'))

                params = lda.components_
                params = pd.DataFrame(params)
                if filt>0:
                    params.columns = keep_columns
                else:
                    params.columns = kmer_list

                thisdic = {}
                if filt>0:
                    for km in keep_columns:
                        thisdic[km] = list(params[km])
                else:
                    for km in kmer_list:
                        thisdic[km] = list(params[km])
                pickle.dump(thisdic,open(f'{save_dir}/{opt.seq_type}LDA{k}_probabilities_dict_lda{nb_lda_components}.p','wb'))


if __name__ == '__main__':
        main()
