"""
Articles module
"""

import re

from nltk import sent_tokenize
from datasets import load_dataset


class Articles:
    """
    Streams Wikipedia article abstracts from a dataset.
    """

    def __init__(self, dataset):
        """
        Creates an article stream.

        Args:
            dataset: dataset path
        """

        # Load the raw dataset
        self.wiki = load_dataset(dataset, split="train")

    def __call__(self):
        """
        Streams valid Wikipedia article abstracts.
        """

        # Titles and abstract prefixes to skip
        skiptitles = ["List of ", "Timeline of ", "Timelines "]
        skipabstracts = ["Events ", "The following events ", "REDIRECT "]

        for row in self.wiki:
            # Article title
            title = row["title"]

            # First text section is considered abstract
            abstract = self.abstract(row["text"])

            # Accept article using following rules
            # - title does not contain 'disambiguation' and does not start with skip title prefixes
            # - abstract is not empty and does not end with ':'  and does not start with skip abstract prefixes
            # - lede does not contain 'can refer to' or 'may refer to'
            if (
                "disambiguation" not in title
                and not any(title.startswith(p) for p in skiptitles)
                and abstract and not abstract.endswith(":")
                and not any(abstract.startswith(p) for p in skipabstracts)
            ):
                # Split into sentences
                lede = sent_tokenize(abstract)[0]

                # Skip if lede is a list of references
                if lede.strip() and "can refer to" not in lede and "may refer to" not in lede:
                    yield (title, abstract)

    def __len__(self):
        """
        Returns the dataset length.
        """

        return len(self.wiki)

    def abstract(self, text):
        """
        Builds an abstract from article text. The first paragraph with text is used as the abstract.

        Args:
            text: article text

        Returns:
            abstract text
        """

        # Detect and remove formatting boxes
        formatbox = ("{", "}", "|", "[", "]", "!")
        if text.strip().startswith(formatbox):
            # Remove info boxes
            text = re.sub(r"\{[{|\n].*[}|\n]\}", "", text, flags=re.DOTALL)

            # Remove attachment links
            text = re.sub(r"\[.*\]", "", text, flags=re.DOTALL)

            # Filter lines that start with a formatting box character - handles partial formatting boxes
            text = "\n".join(x for x in text.split("\n") if not x.strip().startswith(formatbox))

            # Filter empty and low token sections parsed
            text = "\n\n".join(x for x in text.split("\n\n") if x.strip() and len(x.split()) >= 5)

        # The first/lede section in the article is considered the abstract
        sections = text.split("\n\n")
        abstract = sections[0].strip() if sections else ""

        # Cleanup empty parens
        abstract = re.sub(r" \(\s*\)", "", abstract)
        return abstract
