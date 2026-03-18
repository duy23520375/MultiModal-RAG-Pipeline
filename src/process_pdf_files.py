from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

def partition_documents(file_path):
  print("Partition Documents.....")
  elements = partition_pdf(
    filename=file_path,
    strategy='hi_res',
    infer_table_structure=True,
    extract_image_block_types=["Image"],
    extract_image_block_to_payload=True
  )
  print(f"{len(elements)} elements from {file_path}")
  return elements

def create_chunk_by_title(elements):
  print("Chunking.....")
  chunks = chunk_by_title(
    elements=elements,
    combine_text_under_n_chars=500,
    max_characters=3000,
    new_after_n_chars=2400
  )
  print(f"Created {len(chunks)} chunks")
  return chunks

def process_pdf_files(file_path):
  elements = partition_documents(file_path)
  chunks = create_chunk_by_title(elements)
  print("Chunking Successfully!!!")
  return chunks
  