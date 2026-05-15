from datetime import datetime

from data.utils.data_utils import date_chunks


def test_date_chunks_covers_full_range_contiguously():
    start, end = datetime(2020, 1, 1), datetime(2022, 12, 31)
    chunks = list(date_chunks(start, end, max_days=366))
    assert chunks[0][0] == start
    assert chunks[-1][1] == end
    # Chunks are contiguous and non-overlapping (each starts the day after the
    # previous one ends).
    for (_, chunk_end), (next_start, _) in zip(chunks, chunks[1:]):
        assert (next_start - chunk_end).days == 1
    # No chunk exceeds max_days.
    for chunk_start, chunk_end in chunks:
        assert (chunk_end - chunk_start).days <= 365


def test_date_chunks_single_chunk_for_short_range():
    start, end = datetime(2022, 1, 1), datetime(2022, 1, 10)
    assert list(date_chunks(start, end, max_days=366)) == [(start, end)]


def test_date_chunks_handles_same_day():
    day = datetime(2022, 6, 1)
    assert list(date_chunks(day, day, max_days=366)) == [(day, day)]
