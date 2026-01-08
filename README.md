<h1>README</h1>
After downloading the code, you need to install the necessary libraries.

You can do this by running the line:

```pip install -r requirements.txt```

After doing that, change the directory to the code file, or open the terminal on your compiler.

Then, you can run the lines:
```
python main.py --ratings ml-100k/u.data --method mf --n_factors 50 --epochs 15
python main.py --ratings ml-100k/u.data --method ibcf --n_factors 50 --epochs 15
python main.py --ratings ml-1m/ratings.dat --sep "::" --method mf --similarity cosine --k 40
python main.py --ratings ml-1m/ratings.dat --sep "::" --method ibcf --similarity cosine --k 40
```

Doing this will run the code for Matrix Factorization, and Item-Based Collaborative Filtering 
using either the MovieLens 100K dataset, or the MovieLens 1M dataset; depending on which line you run.
