# Data Preparation Guide

## Main Workflow

1. Download the dataset from the link provided in the table below.
2. Unzip the dataset and put it in the **Any** directory.
    - Make sure all data files are in the same directory.
    - Necessary files for Yelp:
        - `yelp_academic_dataset_business.json`, 
        - `yelp_academic_dataset_user.json`, 
        - `yelp_academic_dataset_review.json`
    - Necessary files for Amazon:
        - `Industrial_and_Scientific.csv`, 
        - `Musical_Instruments.csv`, 
        - `Video_Games.csv`,
        - `Industrial_and_Scientific.jsonl`, 
        - `Musical_Instruments.jsonl`, 
        - `Video_Games.jsonl`,
        - `meta_Industrial_and_Scientific.jsonl`, 
        - `meta_Musical_Instruments.jsonl`, 
        - `meta_Video_Games.jsonl`
    - Necessary files for Goodreads:
        - `goodreads_books_children.json`, 
        - `goodreads_reviews_children.json`, 
        - `goodreads_books_comics_graphic.json`, 
        - `goodreads_reviews_comics_graphic.json`, 
        - `goodreads_books_poetry.json`, 
        - `goodreads_reviews_poetry.json`
3. Run the `data_process.py` script to prepare the data for the simulation and recommendation tasks.
```bash
python data_process.py --input <path_to_raw_dataset> --output <path_to_processed_dataset>
```

## Dataset Overview and Download Links

|                                | len(review)   | len(business) | len(user)   | link                                                         |
| ------------------------------ | ------------- | ------------- | ----------- | ------------------------------------------------------------ |
| **Yelp**                       | **-** | **-**    | **-** | [download](https://www.yelp.com/dataset)                                                             |
| -                              | -             | -             | -           |                                                              |
| Industrial_and_Scientific      | 412.9K        | 25.8K         | 51.0K       | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Industrial_and_Scientific.jsonl.gz),  [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Industrial_and_Scientific.jsonl.gz), [rating_only](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/benchmark/5core/rating_only/Industrial_and_Scientific.csv.gz) |
| Musical_Instruments            | 511.8K        | 24.6K         | 57.4K       | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Musical_Instruments.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Musical_Instruments.jsonl.gz), [rating_only](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/benchmark/5core/rating_only/Musical_Instruments.csv.gz) |
| Video_Games                    | 814.6K        | 25.6K         | 94.8K       | [review](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Video_Games.jsonl.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Video_Games.jsonl.gz), [rating_only](https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/benchmark/5core/rating_only/Video_Games.csv.gz) |
| **Amazon**                     | **1,739,300** | **76,000**    |             |                                                              |
| -                              | -             | -             | -           |                                                              |
| Goodreads<br />_Children       | 734,640       | 124,082       |             | [review](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_books_children.json.gz) |
| Goodreads<br />_Comics&Graphic | 542,338       | 89,411        |             | [review](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_books_comics_graphic.json.gz) |
| Goodreads<br />_Poetry         | 154,555       | 36,514        |             | [review](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz), [meta](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/byGenre/goodreads_books_poetry.json.gz) |
| **Goodreads**                  | **1,431,533** | **250,007**   |             |                                                              |