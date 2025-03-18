# Biomedical Document Clustering Tool

This tool is designed specifically for clustering biomedical research papers without requiring predefined categories. It employs natural language processing and machine learning techniques to automatically group similar papers based on their content.

## Features

- **Specialized for Biomedical Papers**: Optimized for PMC (PubMed Central) and similar formats
- **Multiple Clustering Algorithms**: K-Means, DBSCAN, and Hierarchical Clustering
- **Biomedical Embeddings**: Option to use domain-specific BERT models
- **Interactive Visualizations**: View paper clusters using t-SNE and UMAP
- **Automatic Optimal Clustering**: Finds the ideal number of clusters
- **Key Term Extraction**: Identifies important terms that characterize each cluster

## Setup Instructions

1. Place your biomedical papers in a directory called `pmc_data` in the same location as the scripts.

2. Install the required packages:
   ```bash
   pip install numpy pandas matplotlib scikit-learn sentence-transformers umap-learn nltk seaborn
   ```

3. For biomedical-specific embeddings (option 3), install the additional model:
   ```bash
   pip install transformers
   ```

## Usage

Run the demo script:
```bash
python demo.py
```

The script will analyze the papers in your `pmc_data` folder and present you with three options:

1. **Run K-Means Clustering**: Uses the K-Means algorithm with automatic optimal cluster detection
2. **Compare All Clustering Methods**: Runs and compares K-Means, DBSCAN, and Hierarchical clustering
3. **Use Biomedical Embedding Model**: Applies the specialized PubMedBERT model for improved biomedical text understanding

## Understanding the Output

After running the clustering, you'll get several outputs:

### Visual Outputs
- **optimal_k_silhouette.png**: Graph showing how the optimal number of clusters was determined
- **tsne_visualization.png**: 2D representation of document clusters using t-SNE
- **umap_visualization.png**: Alternative visualization using UMAP
- **cluster_composition_*.png**: Distribution of documents across clusters (if metadata available)

### Console Outputs
- **Cluster Summary**: Shows the number of documents in each cluster
- **Key Terms**: Most representative terms for each cluster
- **Sample Documents**: Examples from each cluster to help understand the grouping

## Customizing for Your Papers

The tool is designed to work out-of-the-box with most biomedical papers, but you can adjust settings for specific needs:

### In `document_clustering.py`:
- **Preprocessing**: Modify `preprocess_documents` parameters:
  - `keep_numbers=True`: Keep or remove numerical data
  - `max_length=20000`: Adjust document truncation length

- **Embedding Model**: Change in the initialization:
  ```python
  doc_clustering = DocumentClustering(
      documents_path="./pmc_data",
      embedding_model="pritamdeka/S-PubMedBert-MS-MARCO"  # Change to a different model if needed
  )
  ```

- **Clustering Parameters**:
  - K-Means: Adjust `n_clusters` 
  - DBSCAN: Modify `eps` and `min_samples`
  - Hierarchical: Change `linkage` parameter

## Additional Notes

### Metadata
If you have metadata about your papers (e.g., categories, journals, authors), you can create a `metadata.csv` file in the `pmc_data` folder with columns for `document_id` and `category`. The tool will use this to evaluate clustering quality.

The tool can also attempt to extract basic metadata from your papers if no metadata file exists.

### Large Document Collections
For very large collections (hundreds or thousands of papers):
- Consider increasing `max_length` parameter in preprocessing to capture more content
- Use the biomedical-specific embedding model (option 3) for better semantic understanding
- Run the comparison (option 2) first to determine which clustering algorithm works best

### Memory Usage
Processing large documents can be memory-intensive. If you encounter memory issues:
- Reduce `max_length` in the preprocessing step
- Process a smaller subset of papers initially

## Troubleshooting

- **XML parsing errors**: The tool will fall back to processing raw text if XML parsing fails
- **Empty/short documents**: The pipeline automatically handles and filters these
- **Embedding model errors**: If the biomedical model fails to load, it will revert to the default model
- **Visualization issues**: For very large or very small document sets, visualization parameters are automatically adjusted