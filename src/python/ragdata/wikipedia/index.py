"""
Index module
"""

import argparse
import csv
import logging
import sqlite3

from ..base import Index as IndexBase
from ..base import Reader as ReaderBase
from ..base import COMPLETE, BATCH, ENCODEBATCH

from .articles import Articles


class Reader(ReaderBase):
    """
    Loads the Wikipedia dataset along with the page view database and adds valid entries to the outputs queue.
    """

    def __call__(self, outputs, args):
        """
        Adds valid Wikipedia articles to outputs.

        Args:
            outputs: outputs queue
            args: command line args
        """

        # Streams Wikipedia titles and abstracts
        wiki = Articles(args.dataset)

        # Get percentile rankings
        rank = self.rankings(args.pageviews)

        # Domain labels
        labels = self.labels(args.labels)

        # Put estimated data size
        outputs.put(len(wiki))

        # Batch of rows
        batch = []

        for title, abstract in wiki():
            score = self.percentile(rank, title)

            # Index article
            batch = self.add(batch, {
                "id": title,
                "text": abstract,
                "percentile": score,
                "domain": labels[title]
            }, outputs)

        # Final batch
        if batch:
            outputs.put(batch)

        # Complete flag
        outputs.put(COMPLETE)

    def rankings(self, path):
        """
        Reads a page views database at path and runs a query to rank each article by page
        view percentile.

        Args:
            path: path to database file

        Returns:
            dictionary of title to percentile rank for each article
        """

        # Read page views database
        connection = sqlite3.connect(path)
        cursor = connection.cursor()

        # Get ranks for each page
        cursor.execute("SELECT title, percent_rank() OVER (ORDER BY views) rank FROM pages")

        rank = {}
        for title, score in cursor:
            rank[title] = score

        return rank

    def percentile(self, rank, title):
        """
        Looks up the percentile for a title.

        Args:
            rank: ranking dictionary
            title: title key to lookup

        Returns:
            percentile rank for title, 0 if not found
        """

        return rank.get(title.lower().replace(" ", "_"), 0)

    def labels(self, path):
        """
        Reads a labels csv as a dictionary.

        Args:
            path: path to csv file
        """

        with open(path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Return dict of {id: label}
            return {row["id"]: row["label"] for row in reader}


class Index(IndexBase):
    """
    Builds a Wikipedia embeddings index.
    """

    def __init__(self):
        """
        Sets the embeddings configuration.
        """

        # Call parent constructor
        super().__init__()

        # Create configuration
        self.config = {
            "format": "json",
            "path": "intfloat/e5-base",
            "instructions": {"query": "query: ", "data": "passage: "},
            "batch": BATCH,
            "encodebatch": ENCODEBATCH,
            "faiss": {"quantize": True, "sample": 0.05},
            "content": True,
            "columns": {"store": ["percentile", "domain"]},
            "expressions": [
                {"name": "percentile", "index": True},
                {"name": "domain", "index": True},
            ]
        }

        # Create Reader instance
        self.reader = Reader()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(funcName)s: %(message)s")
    logging.getLogger().setLevel(logging.INFO)

    # Command line parser
    parser = argparse.ArgumentParser(description="Wikipedia Index")
    parser.add_argument("-d", "--dataset", help="input dataset", metavar="DATASET", required=True)
    parser.add_argument("-l", "--labels", help="path to labels csv", metavar="LABELS", required=True)
    parser.add_argument("-o", "--output", help="path to output directory", metavar="OUTPUT", required=True)
    parser.add_argument("-v", "--pageviews", help="path to pageviews database", metavar="PAGEVIEWS", required=True)

    # Build index
    index = Index()
    index(parser.parse_args())
