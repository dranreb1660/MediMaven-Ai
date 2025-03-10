@step
def generate_embeddings(
    df: pd.DataFrame,
    batch_size: int = 1,
    device_name: str = "cpu"
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Step C: Use a Transformer model to embed each context passage.

    :param df: DataFrame with "context_id" and "context".
    :param batch_size: Batch size for encoding.
    :param device_name: Device to use for encoding ("cpu" or "mps" or "cuda").
    :return: (df, embeddings)
        df: the same DataFrame (for reference)
        embeddings: a numpy array of shape (num_contexts, embedding_dim).
    """
    # Set torch to use single thread to avoid parallel processing issues
    torch.set_num_threads(1)
    
    print(f"Use pytorch device_name: {device_name}")
    model = SentenceTransformer(model_name, device=device_name)
    start_time = time.time()

    contexts = df["context"].tolist()
    
    # Custom encoding function with error handling that processes one text at a time
    def encode_with_error_handling(texts_list, model, batch_size=1):
        all_embeddings = []
        total = len(texts_list)
        
        # Process in small batches
        for i in range(0, total, batch_size):
            batch = texts_list[i:i+batch_size]
            try:
                print(f"Processing batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}")
                batch_embeddings = model.encode(batch, show_progress_bar=False)
                all_embeddings.extend(batch_embeddings)
                
                # Explicit memory cleanup after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Free up some memory on any device
                if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                    # For MPS (Apple Silicon)
                    torch.mps.empty_cache()
                
            except Exception as e:
                print(f"Error processing batch starting at index {i}: {str(e)}")
                # If a batch fails, try processing items one by one
                for j, text in enumerate(batch):
                    try:
                        single_embedding = model.encode([text], show_progress_bar=False)
                        all_embeddings.append(single_embedding[0])
                    except Exception as inner_e:
                        print(f"Error processing individual text at index {i+j}: {str(inner_e)}")
                        # Use a zero vector of appropriate dimension as fallback
                        if all_embeddings:
                            # Use the same dimension as successful embeddings
                            all_embeddings.append(np.zeros_like(all_embeddings[0]))
                        else:
                            # If no successful embedding yet, use model's output dimension
                            all_embeddings.append(np.zeros(model.get_sentence_embedding_dimension()))
                
                # Cleanup after error handling
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                
        return np.array(all_embeddings)
    
    # Use our custom encoding function
    embeddings = encode_with_error_handling(contexts, model, batch_size=batch_size)
    embeddings = np.array(embeddings).astype("float32")  # For FAISS, float32 is typical
