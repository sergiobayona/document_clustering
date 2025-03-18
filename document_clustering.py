import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import umap
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.metrics import silhouette_score
import glob
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Ensure NLTK downloads are successful by specifying download directory
nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

print("Checking NLTK resources...")

# Force download punkt regardless of whether it's found
try:
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    print("NLTK punkt downloaded successfully.")
except Exception as e:
    print(f"Warning: Failed to download NLTK punkt: {e}")
    print("You may need to manually download NLTK resources.")

# Force download stopwords
try:
    nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
    print("NLTK stopwords downloaded successfully.")
except Exception as e:
    print(f"Warning: Failed to download NLTK stopwords: {e}")
    print("You may need to manually download NLTK resources.")

# Add paths to NLTK data directories
nltk.data.path.append(nltk_data_dir)

class DocumentClustering:
    def __init__(self, documents_path, embedding_model="all-MiniLM-L6-v2"):
        """
        Initialize the document clustering pipeline.
        
        Args:
            documents_path: Path to the folder containing text documents
            embedding_model: Name of the SentenceTransformer model to use
        """
        self.documents_path = documents_path
        self.documents = []
        self.doc_names = []
        self.embeddings = None
        self.clusters = None
        self.cluster_method = None
        self.cluster_labels = None
        
        # Initialize the embedding model with error handling
        try:
            self.model = SentenceTransformer(embedding_model)
        except Exception as e:
            print(f"Error initializing SentenceTransformer model: {e}")
            print("Make sure the model exists and you have an internet connection.")
            raise
        
    def load_documents(self, file_extension=None):
        """
        Load all documents from the specified path.
        
        Args:
            file_extension: File extension to filter by (e.g., ".txt", ".xml")
                           If None, load all files.
        """
        if file_extension:
            files = glob.glob(os.path.join(self.documents_path, f"*{file_extension}"))
        else:
            # Get all files in the directory
            files = [os.path.join(self.documents_path, f) for f in os.listdir(self.documents_path) 
                    if os.path.isfile(os.path.join(self.documents_path, f))]
        
        self.doc_names = [os.path.basename(f) for f in files]
        
        total_files = len(files)
        loaded_files = 0
        
        print(f"Found {total_files} files. Loading...")
        
        for file in files:
            try:
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                    # Check if it's potentially XML
                    if content.strip().startswith('<?xml') or content.strip().startswith('<'):
                        try:
                            import xml.etree.ElementTree as ET
                            # Try to parse as XML and extract text content
                            root = ET.fromstring(content)
                            
                            # Extract article title
                            title_elem = root.find('.//article-title')
                            title = title_elem.text if title_elem is not None and hasattr(title_elem, 'text') else ""
                            
                            # Extract abstract
                            abstract_elems = root.findall('.//abstract//p')
                            abstract = ' '.join([p.text for p in abstract_elems if p.text is not None]) if abstract_elems else ""
                            
                            # Extract body text
                            body_elems = root.findall('.//body//p')
                            body = ' '.join([p.text for p in body_elems if p.text is not None]) if body_elems else ""
                            
                            # Combine elements
                            text = f"{title} {abstract} {body}"
                            if text.strip():  # Only add if we extracted some text
                                self.documents.append(text)
                            else:
                                # If we couldn't extract text, fall back to using the full content
                                self.documents.append(content)
                        except Exception as e:
                            # If XML parsing fails, fall back to using the full content
                            self.documents.append(content)
                    else:
                        # Not XML, extract sections for biomedical papers
                        # Look for common section headers in biomedical papers
                        abstract_match = re.search(r'Abstract\s*\n(.*?)(?:\n\s*\n|\n[A-Z][A-Za-z\s]+\n)', content, re.DOTALL)
                        abstract = abstract_match.group(1) if abstract_match else ""
                        
                        # Get introduction and methods if available
                        intro_match = re.search(r'INTRODUCTION\s*\n(.*?)(?:\n\s*\n|\n[A-Z][A-Za-z\s]+\n)', content, re.DOTALL)
                        intro = intro_match.group(1) if intro_match else ""
                        
                        # Just use the full content, but prioritize abstract and introduction
                        text = f"{abstract} {intro} {content}"
                        self.documents.append(text)
                    
                    loaded_files += 1
            except Exception as e:
                print(f"Error loading file {file}: {e}")
        
        print(f"Successfully loaded {loaded_files} out of {total_files} documents.")
        return self
    
    def preprocess_documents(self, keep_numbers=True, max_length=20000):
        """
        Preprocess documents with cleaning optimized for scientific papers.
        
        Args:
            keep_numbers: Whether to keep numbers (important for biomedical papers)
            max_length: Maximum document length to process (for memory efficiency)
        """
        processed_docs = []
        
        # Check if stopwords are available, if not use a minimal set
        try:
            stop_words = set(stopwords.words('english'))
            # Remove potentially important scientific words from stopwords
            scientific_terms = {'between', 'through', 'during', 'before', 'after', 
                              'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 
                              'over', 'under', 'again', 'further', 'then', 'once', 'here',
                              'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                              'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
                              'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'}
            for term in scientific_terms:
                if term in stop_words:
                    stop_words.remove(term)
        except:
            print("Warning: Unable to load stopwords. Using a minimal set.")
            # Minimal set of common English stopwords if NLTK fails
            stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'is', 'are', 'was', 'were', 
                         'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'to', 
                         'from', 'for', 'with', 'this', 'that', 'these', 'those', 'of', 'at', 'by'}
        
        # Create a simple tokenizer function to use as fallback
        def simple_tokenize(text):
            # Remove punctuation and split on whitespace
            text = re.sub(r'[^\w\s]', ' ', text)
            # Split by whitespace and keep only non-empty tokens
            return [token for token in text.split() if token]
        
        for doc in self.documents:
            try:
                # Truncate very long documents to prevent memory issues
                doc = doc[:max_length]
                
                # Convert to lowercase
                doc = doc.lower()
                
                # Handle special characters based on setting
                if keep_numbers:
                    # Keep alphanumeric characters, replace others with space
                    doc = re.sub(r'[^a-zA-Z0-9\s]', ' ', doc)
                else:
                    # Remove special characters and numbers
                    doc = re.sub(r'[^a-zA-Z\s]', ' ', doc)
                    
                # Remove extra whitespace
                doc = re.sub(r'\s+', ' ', doc).strip()
                
                # Try to use NLTK's word_tokenize first
                tokenize_success = False
                try:
                    tokens = word_tokenize(doc)
                    tokenize_success = True
                except Exception as e:
                    # If NLTK tokenization fails, try to download punkt again
                    try:
                        nltk.download('punkt', quiet=True)
                        tokens = word_tokenize(doc)
                        tokenize_success = True
                    except:
                        tokenize_success = False
                
                # If NLTK tokenization still fails, use simple tokenization
                if not tokenize_success:
                    tokens = simple_tokenize(doc)
                    if not tokens and doc:  # Empty tokens but non-empty doc
                        # If tokenization produced nothing but we had content, keep the original
                        print(f"Warning: Tokenization produced no tokens for non-empty document. Using raw text.")
                        tokens = doc.split()
                
                # Remove stopwords
                filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
                
                # Only keep tokens with length > 1 (skip single characters)
                filtered_tokens = [token for token in filtered_tokens if len(token) > 1]
                
                # If we lost all tokens, try to preserve some content
                if not filtered_tokens and tokens:
                    filtered_tokens = [token for token in tokens if len(token) > 1]
                
                processed_doc = ' '.join(filtered_tokens)
                processed_docs.append(processed_doc)
            except Exception as e:
                print(f"Error preprocessing document: {e}")
                # Add empty string as a fallback to maintain document count
                processed_docs.append("")
            
        self.documents = processed_docs
        print(f"Preprocessed {len(processed_docs)} documents (keeping numbers: {keep_numbers})")
        return self
    
    def generate_embeddings(self):
        """Generate embeddings for all documents using SentenceTransformer."""
        print("Generating embeddings...")
        # Handle empty documents
        valid_docs = [doc for doc in self.documents if doc.strip()]
        if len(valid_docs) < len(self.documents):
            print(f"Warning: {len(self.documents) - len(valid_docs)} empty documents were skipped.")
            
            # Update document_names to match the valid documents
            valid_indices = [i for i, doc in enumerate(self.documents) if doc.strip()]
            self.doc_names = [self.doc_names[i] for i in valid_indices]
            self.documents = valid_docs
            
        # Generate embeddings for valid documents
        if valid_docs:
            self.embeddings = self.model.encode(valid_docs, show_progress_bar=True)
            print(f"Generated embeddings with shape: {self.embeddings.shape}")
        else:
            print("Error: No valid documents to generate embeddings.")
            self.embeddings = np.array([])
            
        return self
    
    def cluster_kmeans(self, n_clusters=5):
        """Perform K-means clustering on document embeddings."""
        if self.embeddings is None or len(self.embeddings) == 0:
            raise ValueError("Embeddings not generated or empty. Call generate_embeddings() first.")
        
        print(f"Running K-Means clustering with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.clusters = kmeans.fit_predict(self.embeddings)
        
        # Calculate silhouette score for cluster quality evaluation
        if n_clusters > 1 and len(self.embeddings) > n_clusters:
            silhouette_avg = silhouette_score(self.embeddings, self.clusters)
            print(f"Silhouette Score: {silhouette_avg:.4f}")
        
        self.cluster_method = "K-means"
        self.cluster_labels = self.clusters
        return self
    
    def cluster_dbscan(self, eps=0.5, min_samples=5):
        """Perform DBSCAN clustering on document embeddings."""
        if self.embeddings is None or len(self.embeddings) == 0:
            raise ValueError("Embeddings not generated or empty. Call generate_embeddings() first.")
        
        print(f"Running DBSCAN clustering with eps={eps}, min_samples={min_samples}...")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.clusters = dbscan.fit_predict(self.embeddings)
        
        # Count the number of clusters (excluding noise points labeled as -1)
        n_clusters = len(set(self.clusters)) - (1 if -1 in self.clusters else 0)
        print(f"DBSCAN found {n_clusters} clusters and {list(self.clusters).count(-1)} noise points.")
        
        if n_clusters > 1:  # Calculate silhouette score only if more than one cluster is found
            # Filter out noise points for silhouette score calculation
            mask = self.clusters != -1
            if sum(mask) > n_clusters:  # Need more samples than clusters
                silhouette_avg = silhouette_score(self.embeddings[mask], self.clusters[mask])
                print(f"Silhouette Score (excluding noise): {silhouette_avg:.4f}")
        
        self.cluster_method = "DBSCAN"
        self.cluster_labels = self.clusters
        return self
    
    def cluster_hierarchical(self, n_clusters=5, linkage='ward'):
        """Perform hierarchical clustering on document embeddings."""
        if self.embeddings is None or len(self.embeddings) == 0:
            raise ValueError("Embeddings not generated or empty. Call generate_embeddings() first.")
        
        print(f"Running Hierarchical clustering with {n_clusters} clusters, linkage={linkage}...")
        hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        self.clusters = hc.fit_predict(self.embeddings)
        
        # Calculate silhouette score for cluster quality evaluation
        if n_clusters > 1 and len(self.embeddings) > n_clusters:
            silhouette_avg = silhouette_score(self.embeddings, self.clusters)
            print(f"Silhouette Score: {silhouette_avg:.4f}")
        
        self.cluster_method = "Hierarchical"
        self.cluster_labels = self.clusters
        return self
    
    def find_optimal_k(self, max_k=15):
        """Find optimal number of clusters for K-means using silhouette scores."""
        if self.embeddings is None or len(self.embeddings) == 0:
            raise ValueError("Embeddings not generated or empty. Call generate_embeddings() first.")
            
        # Make sure max_k is valid
        if max_k < 2:
            print("Warning: max_k must be at least 2. Setting to 2.")
            max_k = 2
            
        # Limit max_k to number of samples - 1
        max_k = min(max_k, len(self.embeddings) - 1)
            
        print("Finding optimal number of clusters...")
        silhouette_scores = []
        
        for k in range(2, max_k+1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.embeddings)
            silhouette_avg = silhouette_score(self.embeddings, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            print(f"K={k}: Silhouette Score={silhouette_avg:.4f}")
        
        # Plot silhouette scores
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, max_k+1), silhouette_scores, 'o-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score Method For Optimal k')
        plt.grid(True)
        plt.savefig('optimal_k_silhouette.png')
        
        # Find optimal k (highest silhouette score)
        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
        print(f"Optimal number of clusters: {optimal_k}")
        return optimal_k
    
    def visualize_clusters_tsne(self, perplexity=30, random_state=42):
        """Visualize clusters using t-SNE for dimensionality reduction."""
        if self.clusters is None:
            raise ValueError("Clusters not generated. Run clustering first.")
            
        if len(self.embeddings) < 2:
            print("Warning: Not enough samples for t-SNE visualization.")
            return None
        
        # Apply t-SNE
        print("Applying t-SNE for visualization...")
        tsne = TSNE(n_components=2, perplexity=min(perplexity, len(self.embeddings) - 1), 
                   random_state=random_state)
        embeddings_2d = tsne.fit_transform(self.embeddings)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'cluster': self.clusters,
            'document': self.doc_names if self.doc_names else range(len(self.clusters))
        })
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.title(f'Document Clusters Visualization using t-SNE\nClustering method: {self.cluster_method}')
        
        # If DBSCAN was used and there are noise points
        if self.cluster_method == "DBSCAN" and -1 in self.clusters:
            # Plot noise points first
            noise = df[df['cluster'] == -1]
            sns.scatterplot(data=noise, x='x', y='y', color='gray', alpha=0.5, label='Noise')
            
            # Then plot clustered points
            clustered = df[df['cluster'] != -1]
            sns.scatterplot(data=clustered, x='x', y='y', hue='cluster', palette='viridis')
        else:
            sns.scatterplot(data=df, x='x', y='y', hue='cluster', palette='viridis')
        
        plt.savefig('tsne_visualization.png')
        plt.close()
        return embeddings_2d
    
    def visualize_clusters_umap(self, n_neighbors=15, min_dist=0.1, random_state=42):
        """Visualize clusters using UMAP for dimensionality reduction."""
        if self.clusters is None:
            raise ValueError("Clusters not generated. Run clustering first.")
            
        if len(self.embeddings) < 2:
            print("Warning: Not enough samples for UMAP visualization.")
            return None
            
        # Adjust n_neighbors if necessary
        n_neighbors = min(n_neighbors, len(self.embeddings) - 1)
        
        # Apply UMAP
        print("Applying UMAP for visualization...")
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
        embeddings_2d = reducer.fit_transform(self.embeddings)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'cluster': self.clusters,
            'document': self.doc_names if self.doc_names else range(len(self.clusters))
        })
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.title(f'Document Clusters Visualization using UMAP\nClustering method: {self.cluster_method}')
        
        # If DBSCAN was used and there are noise points
        if self.cluster_method == "DBSCAN" and -1 in self.clusters:
            # Plot noise points first
            noise = df[df['cluster'] == -1]
            sns.scatterplot(data=noise, x='x', y='y', color='gray', alpha=0.5, label='Noise')
            
            # Then plot clustered points
            clustered = df[df['cluster'] != -1]
            sns.scatterplot(data=clustered, x='x', y='y', hue='cluster', palette='viridis')
        else:
            sns.scatterplot(data=df, x='x', y='y', hue='cluster', palette='viridis')
        
        plt.savefig('umap_visualization.png')
        plt.close()
        return embeddings_2d
    
    def extract_key_terms(self, cluster_id, top_n=10):
        """Extract key terms that characterize a specific cluster."""
        if self.clusters is None:
            raise ValueError("Clusters not generated. Run clustering first.")
            
        # Get documents in this cluster
        cluster_docs_indices = [i for i, c in enumerate(self.clusters) if c == cluster_id]
        if not cluster_docs_indices:
            print(f"No documents found in cluster {cluster_id}")
            return []
            
        cluster_docs = [self.documents[i] for i in cluster_docs_indices]
        
        # Use TF-IDF to find important terms
        vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = vectorizer.fit_transform(self.documents)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculate average TF-IDF score for each term in this cluster
        cluster_tfidf = tfidf_matrix[cluster_docs_indices].toarray().mean(axis=0)
        
        # Get top terms
        top_term_indices = cluster_tfidf.argsort()[-top_n:][::-1]
        top_terms = [feature_names[i] for i in top_term_indices]
        
        return top_terms
    
    def get_cluster_summary(self):
        """Get a summary of document clusters with sample documents from each cluster."""
        if self.clusters is None:
            raise ValueError("Clusters not generated. Run clustering first.")
        
        print(f"\nCluster Summary ({self.cluster_method}):")
        cluster_summary = {}
        
        # Get unique clusters
        unique_clusters = sorted(set(self.clusters))
        
        for cluster in unique_clusters:
            # Get indices of documents in this cluster
            indices = [i for i, c in enumerate(self.clusters) if c == cluster]
            
            # Get document names for this cluster
            doc_names = [self.doc_names[i] if self.doc_names else f"Document_{i}" for i in indices]
            
            # Create cluster label
            cluster_label = f"Cluster {cluster}" if cluster != -1 else "Noise Points"
            
            # Extract key terms for this cluster
            try:
                key_terms = self.extract_key_terms(cluster, top_n=5)
            except:
                key_terms = []
            
            # Store in summary
            cluster_summary[cluster_label] = {
                'count': len(indices),
                'document_names': doc_names,
                'sample_indices': indices[:3],  # Sample up to 3 documents
                'key_terms': key_terms
            }
            
            # Print summary
            print(f"{cluster_label}: {len(indices)} documents")
            if key_terms:
                print(f"  Key terms: {', '.join(key_terms)}")
            
            # Get a few sample documents to help identify cluster themes
            if indices:
                print("  Sample document names:")
                for i in indices[:3]:  # Show up to 3 samples
                    doc_name = self.doc_names[i] if self.doc_names else f"Document_{i}"
                    print(f"  - {doc_name}")
                print("  Sample content (first 100 chars):")
                for i in indices[:1]:  # Show just 1 sample content
                    print(f"  - {self.documents[i][:100]}...")
                    
        return cluster_summary
    
    def run_pipeline(self, clustering_method='kmeans', n_clusters=5, visualize_method='both'):
        """Run the complete document clustering pipeline."""
        # Load and preprocess documents
        self.load_documents()
        self.preprocess_documents()
        
        # Generate embeddings
        self.generate_embeddings()
        
        # Check if we have enough documents
        if len(self.embeddings) < 2:
            print("Error: Not enough valid documents to perform clustering.")
            return self
            
        # Find optimal number of clusters if using kmeans or hierarchical
        if clustering_method in ['kmeans', 'hierarchical'] and len(self.embeddings) > 2:
            optimal_k = self.find_optimal_k(max_k=min(15, len(self.embeddings)-1))
            n_clusters = optimal_k
        
        # Perform clustering
        if clustering_method == 'kmeans':
            self.cluster_kmeans(n_clusters=n_clusters)
        elif clustering_method == 'dbscan':
            self.cluster_dbscan()
        elif clustering_method == 'hierarchical':
            self.cluster_hierarchical(n_clusters=n_clusters)
        else:
            raise ValueError("Invalid clustering method. Choose 'kmeans', 'dbscan', or 'hierarchical'.")
        
        # Visualize clusters
        if visualize_method in ['tsne', 'both']:
            self.visualize_clusters_tsne()
        if visualize_method in ['umap', 'both']:
            self.visualize_clusters_umap()
        
        # Get cluster summary
        self.get_cluster_summary()
        
        return self

# Example usage
if __name__ == "__main__":
    # Initialize document clustering
    doc_clustering = DocumentClustering(documents_path="./pmc_data")
    
    # Option 1: Run the full pipeline with default settings
    doc_clustering.run_pipeline(clustering_method='kmeans', visualize_method='both')
    
    # Option 2: Run the pipeline step-by-step for more control
    # doc_clustering.load_documents()
    # doc_clustering.preprocess_documents()
    # doc_clustering.generate_embeddings()
    # optimal_k = doc_clustering.find_optimal_k()
    # doc_clustering.cluster_kmeans(n_clusters=optimal_k)
    # doc_clustering.visualize_clusters_umap()
    # doc_clustering.get_cluster_summary()