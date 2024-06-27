import muon
import scanpy as sc
import pandas as pd
import anndata as ad
from anndata import AnnData
from scipy.sparse import csr_matrix

def load_atac_seq_data(
        file_path: str, 
        file_type: str = "h5", 
        threshold: float = 0.05, 
        fragments: bool = False
    ) -> AnnData:
    """
    Given a filepath and parameters, parse and extract the file into a formatted dataset
    that contains fragment counts ATAC seq data

    Args:
        file_path: path to the downloaded file for ATAC seq data
        file_type: type of the ATAC seq data file; "h5" if .h5 or "h5ad" if .h5ad
        threshold: fraction of cells required for the peak to remain in the dataset
        fragments: True if the counts are in fragments; False if they are in read counts

    Returns:
        parsed ATAC seq dataset in AnnData object

    """
    adata: AnnData

    if file_type == "h5":
        mdata = muon.read_10x_h5(file_path)
        adata = mdata.mod["atac"]
        filter_atac_threshold(adata, threshold)
        if not fragments:
            convert_reads_to_fragments(adata, file_type)
        extract_gene_ids(adata)
    
    elif file_type == "h5ad":
        adata = ad.read_h5ad(file_path)
        filter_atac_threshold(adata.layers["counts"], threshold)
        if not fragments:
            convert_reads_to_fragments(adata, file_type)
    
    else:
        raise TypeError(f"File type of {file_type} is not supported.")
    
    return adata



def filter_atac_threshold(adata: AnnData, threshold: float):
    """
    Given an adata of ATAC seq data, filter it such that peaks with at least threshold * # 
    cells remain in the dataset 

    Args:
        adata: AnnData object of the ATAC seq data (cells x peaks)
        threshold: fraction of cells required for the peak to remain in the dataset
    """

    print("Shape before threshold filter: ", adata.shape)

    # compute the threshold: threshold of the cells
    min_cells = int(adata.shape[0] * threshold)
    # in-place filtering of regions
    sc.pp.filter_genes(adata, min_cells=min_cells)

    print("Shape after threshold filter: ", adata.shape)



def convert_reads_to_fragments(adata: AnnData, file_type: str):
    """
    Given an adata of ATAC seq data that contains read counts, convert the counts into
    estimated fragment counts by 1) rounding each count to the nearest integer and 2) dividing 
    the counts by 2

    Args:
        adata: AnnData object of the ATAC seq data (cells x peaks)
    """

    def round_to_even_csr(csr_mat):
        # Access the data array of the CSR matrix
        data = csr_mat.data
        odd_data = data % 2 != 0
        data[odd_data] = data[odd_data] + 1
        data = data / 2
        return csr_matrix((data, csr_mat.indices, csr_mat.indptr), shape=csr_mat.shape)

    adata_input = adata.X if file_type == "h5" else adata.layers['counts']

    print("Before converting reads to fragments:")
    print("1s: ", (adata_input == 1).sum())
    print("2s: ", (adata_input == 2).sum())

    adata.layers['fragments'] = round_to_even_csr(adata_input)

    print("After converting reads to fragments:")
    print("1s: ", (adata.layers['fragments'] == 1).sum())
    print("2s: ", (adata.layers['fragments'] == 2).sum())



def extract_gene_ids(adata: AnnData):
    """
    Given an adata of ATAC seq data, extract chromosome, interval start, and interval end data
    and put them into new variables

    Args:
        adata: AnnData object of the ATAC seq data (cells x peaks)
    """

    split_interval = adata.var["gene_ids"].str.split(":", expand=True)
    adata.var["chr"] = split_interval[0]
    split_start_end = split_interval[1].str.split("-", expand=True)
    adata.var["start"] = split_start_end[0].astype(int)
    adata.var["end"] = split_start_end[1].astype(int)
