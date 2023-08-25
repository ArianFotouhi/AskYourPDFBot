# AskYourPDFBot

One challenge of dealing with a vector DB like FAISS, is to set the number of chuncks and chunck overlap. In fact, the full text (whole characters of documents) would be divided into chuncks. Then FAISS receives each chunck and creates a vector for each one accordingly. 

In general, a larger chunk size will result in more accurate embeddings. This is because the embeddings will be based on more text. However, a larger chunk size can also make it more difficult to find similar documents, as the embeddings will be less specific.

A larger chunk overlap will also result in more accurate embeddings. This is because the embeddings will be based on more context. However, a larger chunk overlap can also make it more difficult to find similar documents, as the embeddings will be less unique.
