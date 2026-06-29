"""MovieLens-1M as a (sequential -> next item) source.

The (text, next-text) framing is more natural for conversational data, but
MovieLens-1M is the standard smoke test for causal-embedding methods (CausE,
DICE, MACR, FCSRec). We map it to text by emitting:

    anchor   = "<previous movie title> (genres: <genres>)"
    positive = "<next movie title> (genres: <genres>)"

per user, ordered by timestamp. This is a thin shim — for real causal-rec
work, use the raw IDs and the appropriate splits, not these strings.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..base import BaseDataset, PairExample
from ..io import download_file, extract
from ..registry import register

_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


@register
class MovieLens1M(BaseDataset):
    name = "movielens_1m"
    description = (
        "MovieLens-1M: 6K users, 4K movies, 1M ratings. Used as text-shimmed "
        "(prev movie -> next movie) pairs for causal-rec smoke tests."
    )
    homepage = "https://grouplens.org/datasets/movielens/1m/"
    citation = "Harper & Konstan, TiiS 2015"
    license = "Custom (research-only)"
    splits = ("train",)

    def _root(self) -> Path:
        return self.data_dir / "ml-1m"

    def is_downloaded(self) -> bool:
        return (self._root() / "ratings.dat").exists() and (
            self._root() / "movies.dat"
        ).exists()

    def download(self) -> None:
        if self.is_downloaded():
            return
        zip_path = self.data_dir / "ml-1m.zip"
        download_file(_URL, zip_path)
        extract(zip_path, self.data_dir)

    def _load_movies(self) -> dict[int, tuple[str, str]]:
        movies: dict[int, tuple[str, str]] = {}
        with (self._root() / "movies.dat").open(encoding="latin-1") as f:
            for line in f:
                parts = line.rstrip("\n").split("::")
                if len(parts) != 3:
                    continue
                mid, title, genres = parts
                movies[int(mid)] = (title, genres)
        return movies

    def _iter_examples(self) -> Iterator[PairExample]:
        movies = self._load_movies()
        # Group ratings by user, sort by timestamp.
        from collections import defaultdict

        per_user: dict[int, list[tuple[int, int]]] = defaultdict(list)
        with (self._root() / "ratings.dat").open() as f:
            for line in f:
                u, m, _r, t = line.rstrip("\n").split("::")
                per_user[int(u)].append((int(t), int(m)))
        for u, items in per_user.items():
            items.sort()
            seq = [movies.get(mid) for _, mid in items if mid in movies]
            for i in range(len(seq) - 1):
                a, b = seq[i], seq[i + 1]
                if not a or not b:
                    continue
                yield PairExample(
                    anchor=f"{a[0]} (genres: {a[1]})",
                    positive=f"{b[0]} (genres: {b[1]})",
                    dataset=self.name,
                    metadata={"user_id": u, "step": i},
                )
