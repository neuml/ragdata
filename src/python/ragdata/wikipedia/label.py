"""
Label module
"""

import argparse
import csv
import logging

from multiprocessing import Process, Queue

from tqdm.auto import tqdm
from txtai.pipeline import Labels

from ..base import Reader as ReaderBase
from ..base import COMPLETE

from .articles import Articles


class Reader(ReaderBase):
    """
    Loads the Wikipedia dataset and adds valid entries to the outputs queue.
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

        # Put wiki size
        outputs.put(len(wiki))

        # Batch of rows
        batch = []

        # Add batches of abstracts
        for title, abstract in wiki():
            batch = self.add(batch, (title, abstract), outputs)

        # Final batch
        if batch:
            outputs.put(batch)

        # Complete flag
        outputs.put(COMPLETE)


class Label:
    """
    Generates domain labels for a Wikipedia articles dataset.
    """

    def __init__(self):
        # Article ids
        self.ids = []

        # Create Reader instance
        self.reader = Reader()

    def __call__(self, args):
        # Encoding parameters
        queue = Queue(5)

        # Dataset reader process
        process = Process(target=self.reader, args=(queue, args))
        process.start()

        # Total size
        total = queue.get()

        # Domain labeler
        labels = Labels("neuml/domain-labeler", dynamic=False)

        with open(args.output, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(["id", "label"])

            # Write each row
            for domain in tqdm(
                labels(self.stream(queue), flatten=True, batch_size=64, max_length=512, truncation=True),
                total=total
            ):
                writer.writerow([self.ids.pop(0), domain[0]])

        # Wait for process to finish and close
        process.join()
        process.close()
        queue.close()

    def stream(self, queue):
        """
        Yields articles for labeling from the queue.

        Args:
            queue: queue to read from
        """

        result = queue.get()
        while result != COMPLETE:
            for title, abstract in result:
                self.ids.append(title)
                yield abstract if abstract else title

            result = queue.get()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(funcName)s: %(message)s")
    logging.getLogger().setLevel(logging.INFO)

    # Command line parser
    parser = argparse.ArgumentParser(description="Wikipedia Labels")
    parser.add_argument("-d", "--dataset", help="input dataset", metavar="DATASET", required=True)
    parser.add_argument("-o", "--output", help="path to output file", metavar="OUTPUT", required=True)

    # Label dataset
    label = Label()
    label(parser.parse_args())
