# coding: utf-8

# ## PlotBot5.py
# ----------------------------------------------------------------------------
# The below code is for a Python based Twitter Bot.
# The bot receives tweets via mentions and in turn performs sentiment analysis
# on the most recent twitter account specified in the mention (i.e. A user
# tweets: "@PlotBot Analyze: @CNN" and in turn the code will trigger sentiment
# analysis on the CNN twitter feed). A plot from the sentiment analysis is
# then tweeted to the PlotBot5 twitter feed. Notable libraries used to
# complete this application include: Matplotlib, Pandas, Tweepy,
# TextBlob, and Seaborn.
# ----------------------------------------------------------------------------

# Dependencies
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt
import json
import tweepy
import time
import seaborn as sns

# Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Incorporate matplotlib inline
# get_ipython().magic(u'matplotlib inline')

# Twitter API Keys
consumer_key = "jCuMds8hkjry8JV8JDEuDVH9o"
consumer_secret = "psgKB7nb05kZqoD2ZFPrG78OqbObHySWUEhcLFcZ03qVMlsCwp"
access_token = "814999527451148288-PVho6BBmmcQbSVKOHBt3E5jbPJM6Krl"
access_token_secret = "a30jMaE70P2kefPFOzrfGTlA06okUcifkjJB9g2JWq4Ih"


# Function for Analyzing Tweets
def AnalyzeTweets():

    # Setup Tweepy API Authentication
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

    # ## Grab Most Recent Mention

    # Run a search for the command term
    mentions = api.search(q="@plotbot5 Analyze:")

    # Input Tweet
    words = []

    # Specified Twitter Account
    target_account = ""

    # Grab the most recent command tweet
    command = mentions["statuses"][0]["text"]
    requesting_user = mentions["statuses"][0]["user"]["screen_name"]

    # Split it to determine the target account
    try:
        words = command.split("Analyze:")
        target_account = words[1].strip()

        # Confirm target_account
        print("Target Account: " + target_account)
        print("Requesting User: " + requesting_user)

    except Exception:
        raise

    # ## Confirm Non-Repeat

    # Grab Self Tweets
    tweets = api.user_timeline()

    # Confirm the target account has never been tweeted before
    repeat = False

    for tweet in tweets:
        if target_account in tweet["text"]:
            repeat = True
            print("Sorry. Repeat detected!")

        else:
            continue

    # ## Run Sentiment Analysis

    # If the tweet specifies a unique account run the analysis
    if not (repeat):

        # Create a generic dictionary for holding all tweet information
        tweet_data = {
            "tweet_source": [],
            "tweet_text": [],
            "tweet_date": [],
            "tweet_vader_score": [],
            "tweet_neg_score": [],
            "tweet_pos_score": [],
            "tweet_neu_score": []
        }

        # Grab 500 tweets from the target source
        for x in xrange(25):

            # Grab the tweets
            tweets = api.user_timeline(target_account, page=x)

            # For each tweet store it into the dictionary
            for tweet in tweets:

                # All data is grabbed from the JSON returned by Twitter
                tweet_data["tweet_source"].append(tweet["user"]["name"])
                tweet_data["tweet_text"].append(tweet["text"])
                tweet_data["tweet_date"].append(tweet["created_at"])

                # Run sentiment analysis on each tweet using Vader
                tweet_data["tweet_vader_score"].append(
                    analyzer.polarity_scores(tweet["text"])["compound"])
                tweet_data["tweet_pos_score"].append(analyzer.polarity_scores(
                    tweet["text"])["pos"])
                tweet_data["tweet_neu_score"].append(analyzer.polarity_scores(
                    tweet["text"])["neu"])
                tweet_data["tweet_neg_score"].append(analyzer.polarity_scores(
                    tweet["text"])["neg"])

    if not (repeat):

        # Create a generic dictionary for holding all tweet information
        tweet_data = {
            "tweet_source": [],
            "tweet_text": [],
            "tweet_date": [],
            "tweet_vader_score": [],
            "tweet_neg_score": [],
            "tweet_pos_score": [],
            "tweet_neu_score": []
        }

        # Grab 500 tweets from the target source
        for x in xrange(25):

            # Grab the tweets
            tweets = api.user_timeline(target_account, page=x)

            # For each tweet store it into the dictionary
            for tweet in tweets:

                # All data is grabbed from the JSON returned by Twitter
                tweet_data["tweet_source"].append(tweet["user"]["name"])
                tweet_data["tweet_text"].append(tweet["text"])
                tweet_data["tweet_date"].append(tweet["created_at"])

                # Run sentiment analysis on each tweet using Vader
                tweet_data["tweet_vader_score"].append(
                    analyzer.polarity_scores(tweet["text"])["compound"])
                tweet_data["tweet_pos_score"].append(analyzer.polarity_scores(
                    tweet["text"])["pos"])
                tweet_data["tweet_neu_score"].append(analyzer.polarity_scores(
                    tweet["text"])["neu"])
                tweet_data["tweet_neg_score"].append(analyzer.polarity_scores(
                    tweet["text"])["neg"])

        # Store the final contents into a DataFrame
        tweet_df = pd.DataFrame(tweet_data, columns=["tweet_source",
                                                     "tweet_text",
                                                     "tweet_date",
                                                     "tweet_vader_score",
                                                     "tweet_pos_score",
                                                     "tweet_neu_score",
                                                     "tweet_neg_score"])

        # Visualize the DataFrame
        tweet_df.head()

    if not (repeat):

        # Convert dates (currently strings) into datetimes
        tweet_df["tweet_date"] = pd.to_datetime(tweet_df["tweet_date"])

        # Sort the dataframe by date
        tweet_df.sort_values("tweet_date", inplace=True)
        tweet_df.reset_index(drop=True, inplace=True)

        # Preview the data to confirm data is sorted
        tweet_df.head()

    if not (repeat):

        # Clear Plot
        plt.clf()

        # Build scatter plot for tracking tweet polarity by tweet history
        # Note how a few data munging tricks were used to obtain (-100 -> 0
        # ticks)
        plt.plot(np.arange(-len(tweet_df["tweet_vader_score"]), 0, 1),
                 tweet_df["tweet_vader_score"], marker="o", linewidth=0.5,
                 alpha=0.8, label="%s" % target_account)

        # Incorporate the other graph properties
        plt.title("Sentiment Analysis of Tweets (%s)" % time.strftime("%x"))
        plt.ylabel("Tweet Polarity")
        plt.xlabel("Tweets Ago")
        plt.xlim([-len(tweet_df["tweet_vader_score"]) - 7, 7])
        plt.ylim([-1.05, 1.05])
        plt.grid(True)

        # Create a legend
        lgnd = plt.legend(fontsize="small", mode="Expanded",
                          numpoints=1, scatterpoints=1,
                          loc="upper left", bbox_to_anchor=(1, 1),
                          title="Tweets", labelspacing=0.5)

        # Save the figure (and account for the legend being outside the plot)
        file_path = "analysis/" + target_account + ".png"
        plt.savefig(file_path, bbox_extra_artists=(lgnd, ),
                    bbox_inches='tight')

    # ## Tweet Plot

    if not (repeat):

        # Tweet out the image and mention the user who requested it
        api.update_with_media(file_path,
                              "New Tweet Analysis: %s (Thx @%s!!)" %
                              (target_account, requesting_user))


# Run the Analyze Tweets Function Every 5 minutes
while(True):
    AnalyzeTweets()
    time.sleep(300)
