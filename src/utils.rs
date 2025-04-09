pub fn determine_chunk_size(nrows: usize) -> usize {
    let num_threads = rayon::current_num_threads();

    let min_rows_per_thread = 16;
    let desired_chunks_per_thread = 4;

    let target_total_chunks = num_threads * desired_chunks_per_thread;
    let chunk_size = nrows.div_ceil(target_total_chunks);

    chunk_size.max(min_rows_per_thread)
}
