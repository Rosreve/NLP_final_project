import json
from collections import Counter 
from util import load_data

class DataInspector:
    def __init__(self):
        self.train, self.dev, self.evidence = load_data()
    
    def show_info(self):
        # dataset size infos
        print(f"Number of training claims: {len(self.train)}")
        print(f"Number of development claims: {len(self.dev)}")
        print(f"Number of evidence entries: {len(self.evidence)}\n")

        print("=" * 100 + "\n")

        # label distribution info
        train_label_counts, dev_label_counts = self.count_label_distribution()
        print("Training label distribution:")
        for label, count in train_label_counts.items():
            print(f"{label}: {count}")

        print("\nDevelopment label distribution:")
        for label, count in dev_label_counts.items():
            print(f"{label}: {count}")

        print("=" * 100 + "\n")

        # show example claim and evidence
        print("Example training claim:")
        self._show_example(self.train)
        print("\nExample development claim:")
        self._show_example(self.dev)
        print("\nExample evidence entry:")
        self._show_example(self.evidence)

    def _show_example(self, dataset, num_examples=2):
        # show raw json data for number of examples from the dataset
        for i, (example_id, example_data) in enumerate(dataset.items()):
            if i >= num_examples:
                break
            print(f"example(s) {i+1} from dataset:")
            print(f"ID: {example_id}")
            print(json.dumps(example_data, indent=2))
            print("\n")

    def count_label_distribution(self):
        train_labels = [
            claim.get("claim_label")
            for claim in self.train.values() # train is always a dict, so we can directly use .values()
            if claim.get("claim_label") is not None
        ]
        dev_labels = [
            claim.get("claim_label")
            for claim in self.dev.values() # dev is always a dict, so we can directly use .values()
            if claim.get("claim_label") is not None
        ]

        train_label_counts = Counter(train_labels)
        dev_label_counts = Counter(dev_labels)

        return train_label_counts, dev_label_counts

if __name__ == "__main__":
    inspector = DataInspector()
    inspector.show_info()

