import os
import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
from document_clustering import DocumentClustering

def check_biomedical_papers():
    """Check the pmc_data folder for biomedical papers."""
    data_path = "pmc_data"
    
    if not os.path.exists(data_path):
        print(f"Error: The folder '{data_path}' does not exist.")
        print("Please make sure your biomedical papers are in a folder named 'pmc_data'.")
        sys.exit(1)
    
    files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    
    if len(files) == 0:
        print(f"Error: No files found in the '{data_path}' folder.")
        sys.exit(1)
    
    print(f"Found {len(files)} files in the '{data_path}' folder.")
    
    # Check file extensions
    extensions = {}
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext in extensions:
            extensions[ext] += 1
        else:
            extensions[ext] = 1
    
    print("File types found:")
    for ext, count in extensions.items():
        print(f"  {ext}: {count} files")
    
    return data_path, files

def evaluate_clusters(doc_clustering, data_path):
    """Evaluate clustering results against true categories if available."""
    metadata_path = os.path.join(data_path, "metadata.csv")
    if not os.path.exists(metadata_path):
        print("Metadata file not found. Skipping evaluation.")
        return
    
    try:
        # Load metadata with true categories
        metadata = pd.read_csv(metadata_path)
        
        # Create a dataframe with document names, predicted clusters, and true categories
        evaluation_df = pd.DataFrame({
            'document': doc_clustering.doc_names,
            'predicted_cluster': doc_clustering.clusters
        })
        
        # Prepare document names for matching (strip paths and extensions)
        eval_doc_base = [os.path.splitext(os.path.basename(doc))[0] for doc in evaluation_df['document']]
        meta_doc_base = [os.path.splitext(os.path.basename(doc))[0] for doc in metadata['document_id']]
        
        # Add the cleaned document names to the dataframes
        evaluation_df['doc_base'] = eval_doc_base
        metadata['doc_base'] = meta_doc_base
        
        # Merge with metadata to get true categories
        evaluation_df = evaluation_df.merge(
            metadata, 
            left_on='doc_base', 
            right_on='doc_base', 
            how='left'
        )
        
        # Check if we have category information
        if 'category' not in evaluation_df.columns:
            print("No category information found in metadata. Skipping evaluation.")
            return
            
        # Display cluster composition by true category
        print("\nCluster Composition by True Category:")
        cluster_composition = pd.crosstab(
            evaluation_df['predicted_cluster'], 
            evaluation_df['category'],
            rownames=['Cluster'],
            colnames=['True Category']
        )
        print(cluster_composition)
        
        # Plot cluster composition
        plt.figure(figsize=(12, 8))
        cluster_composition.plot(kind='bar', stacked=True, figsize=(12, 8))
        plt.title('Cluster Composition by True Category')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Documents')
        plt.tight_layout()
        plt.savefig('cluster_composition.png')
        plt.close()
        
        # Calculate purity for each cluster
        purity_scores = []
        for cluster in set(evaluation_df['predicted_cluster']):
            cluster_docs = evaluation_df[evaluation_df['predicted_cluster'] == cluster]
            if len(cluster_docs) > 0 and not cluster_docs['category'].isna().all():
                # Count documents by category in this cluster
                category_counts = cluster_docs['category'].value_counts()
                # Purity is the ratio of the most common category to the total docs in cluster
                purity = category_counts.iloc[0] / len(cluster_docs)
                purity_scores.append(purity)
                print(f"Cluster {cluster} purity: {purity:.4f} (most common category: {category_counts.index[0]})")
        
        if purity_scores:
            print(f"Average cluster purity: {sum(purity_scores)/len(purity_scores):.4f}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

def compare_clustering_methods(documents_path="./pmc_data"):
    """Compare different clustering methods on the same dataset."""
    clustering_methods = [
        {'method': 'kmeans', 'name': 'K-Means'},
        {'method': 'hierarchical', 'name': 'Hierarchical'},
        {'method': 'dbscan', 'name': 'DBSCAN'}
    ]
    
    results = []
    
    for method in clustering_methods:
        print(f"\n\n{'='*50}")
        print(f"Running {method['name']} clustering")
        print(f"{'='*50}")
        
        try:
            doc_clustering = DocumentClustering(documents_path=documents_path)
            doc_clustering.load_documents()
            doc_clustering.preprocess_documents()
            doc_clustering.generate_embeddings()
            
            if len(doc_clustering.embeddings) < 2:
                print(f"Not enough valid documents for {method['name']} clustering. Skipping.")
                continue
            
            if method['method'] == 'kmeans':
                if len(doc_clustering.embeddings) > 2:
                    optimal_k = doc_clustering.find_optimal_k()
                    doc_clustering.cluster_kmeans(n_clusters=optimal_k)
                else:
                    doc_clustering.cluster_kmeans(n_clusters=1)
            elif method['method'] == 'hierarchical':
                if len(doc_clustering.embeddings) > 2:
                    optimal_k = doc_clustering.find_optimal_k()
                    doc_clustering.cluster_hierarchical(n_clusters=optimal_k)
                else:
                    doc_clustering.cluster_hierarchical(n_clusters=1)
            elif method['method'] == 'dbscan':
                doc_clustering.cluster_dbscan()
            
            # Visualize using both methods if we have enough samples
            if len(doc_clustering.embeddings) > 1:
                doc_clustering.visualize_clusters_tsne()
                doc_clustering.visualize_clusters_umap()
            
            # Get cluster summary
            doc_clustering.get_cluster_summary()
            
            # Evaluate against true categories if available
            evaluate_clusters(doc_clustering, documents_path)
            
            # Rename output files to avoid overwriting
            if os.path.exists("tsne_visualization.png"):
                os.rename("tsne_visualization.png", f"tsne_{method['method']}_visualization.png")
            if os.path.exists("umap_visualization.png"):
                os.rename("umap_visualization.png", f"umap_{method['method']}_visualization.png")
            if os.path.exists("cluster_composition.png"):
                os.rename("cluster_composition.png", f"cluster_composition_{method['method']}.png")
        except Exception as e:
            print(f"Error running {method['name']} clustering: {e}")
            import traceback
            traceback.print_exc()

def extract_paper_metadata():
    """
    Try to extract metadata from papers to create a metadata.csv file.
    This is useful when no predefined categories exist.
    """
    data_path = "pmc_data"
    
    if not os.path.exists(data_path):
        print(f"Error: The folder '{data_path}' does not exist.")
        return
        
    files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    
    if len(files) == 0:
        print(f"Error: No files found in the '{data_path}' folder.")
        return
        
    # Create a dataframe to store metadata
    metadata = []
    
    for file in files:
        try:
            file_path = os.path.join(data_path, file)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # Try to extract title and journal
                title_match = re.search(r'Title:\s*(.*?)\n', content)
                title = title_match.group(1) if title_match else "Unknown"
                
                journal_match = re.search(r'([A-Za-z\s]+)\.\s+\d{4}', content)
                journal = journal_match.group(1).strip() if journal_match else "Unknown"
                
                # Add to metadata
                metadata.append({
                    'document_id': file,
                    'title': title,
                    'journal': journal
                })
        except Exception as e:
            print(f"Error extracting metadata from {file}: {e}")
    
    # Save to CSV
    if metadata:
        df = pd.DataFrame(metadata)
        df.to_csv(os.path.join(data_path, "extracted_metadata.csv"), index=False)
        print(f"Created metadata file with {len(metadata)} entries.")
    else:
        print("No metadata could be extracted.")

def run_biomedical_clustering():
    """Run the biomedical clustering pipeline with appropriate settings."""
    # Check the pmc_data folder for biomedical papers
    pmc_data_path, files = check_biomedical_papers()
    
    # Try to extract metadata if no metadata.csv exists
    if not os.path.exists(os.path.join(pmc_data_path, "metadata.csv")):
        try:
            extract_paper_metadata()
        except Exception as e:
            print(f"Could not extract metadata: {e}")
    
    # Ask user which mode to run
    print("\nBiomedical Paper Clustering")
    print("1. Run single clustering method (K-Means)")
    print("2. Compare all clustering methods")
    print("3. Run with biomedical-specific embedding model")
    choice = input("Enter your choice (1, 2, or 3): ")
    
    try:
        if choice == "1":
            # Initialize and run the document clustering pipeline
            doc_clustering = DocumentClustering(documents_path=pmc_data_path)
            doc_clustering.run_pipeline(clustering_method='kmeans', visualize_method='both')
            
            # Evaluate clustering results if metadata is available
            evaluate_clusters(doc_clustering, pmc_data_path)
        
        elif choice == "2":
            # Compare different clustering methods
            compare_clustering_methods(documents_path=pmc_data_path)
        
        elif choice == "3":
            # Use a biomedical-specific embedding model
            print("\nUsing biomedical-specific embedding model...")
            
            try:
                # Initialize with biomedical BERT model
                doc_clustering = DocumentClustering(
                    documents_path=pmc_data_path,
                    embedding_model="pritamdeka/S-PubMedBert-MS-MARCO"
                )
                
                # Run the pipeline
                doc_clustering.load_documents()
                doc_clustering.preprocess_documents(keep_numbers=True, max_length=20000)
                doc_clustering.generate_embeddings()
                
                if len(doc_clustering.embeddings) > 2:
                    optimal_k = doc_clustering.find_optimal_k()
                    doc_clustering.cluster_kmeans(n_clusters=optimal_k)
                else:
                    print("Not enough valid documents for optimal clustering.")
                    if len(doc_clustering.embeddings) > 0:
                        doc_clustering.cluster_kmeans(n_clusters=1)
                    else:
                        print("No valid embeddings were generated. Exiting.")
                        return
                        
                doc_clustering.visualize_clusters_umap()
                doc_clustering.visualize_clusters_tsne()
                doc_clustering.get_cluster_summary()
                
                # Evaluate clustering results if metadata is available
                evaluate_clusters(doc_clustering, pmc_data_path)
            except Exception as e:
                print(f"Error using biomedical embedding model: {e}")
                print("Falling back to default model...")
                doc_clustering = DocumentClustering(documents_path=pmc_data_path)
                doc_clustering.run_pipeline(clustering_method='kmeans', visualize_method='both')
        
        else:
            print("Invalid choice. Using default K-Means clustering.")
            doc_clustering = DocumentClustering(documents_path=pmc_data_path)
            doc_clustering.run_pipeline(clustering_method='kmeans', visualize_method='both')
    except Exception as e:
        print(f"An error occurred during clustering: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_biomedical_clustering()