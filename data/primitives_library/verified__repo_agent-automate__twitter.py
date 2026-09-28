"""Twitter/X integration."""
from __future__ import annotations

from .base import BaseIntegration

API = "https://api.twitter.com/2"


class TwitterIntegration(BaseIntegration):
    name = "twitter"
    label = "Twitter/X"
    env_vars = {
        "TWITTER_BEARER_TOKEN": "Bearer token from developer.twitter.com",
    }

    def _auth(self) -> dict:
        return {
            "Authorization": f"Bearer {self.env('TWITTER_BEARER_TOKEN')}",
        }

    def register(self, mcp) -> None:
        integration = self

        @mcp.tool()
        def twitter_search_tweets(query: str, max_results: int = 10) -> str:
            """
            Search recent tweets (last 7 days).

            Args:
                query: Search query string (e.g. "#python lang:en").
                max_results: Number of results (10-100, default 10).
            """
            import urllib.parse

            params = urllib.parse.urlencode(
                {
                    "query": query,
                    "max_results": max(10, min(max_results, 100)),
                    "tweet.fields": "created_at,author_id,text",
                }
            )
            response = integration.get(
                f"{API}/tweets/search/recent?{params}",
                integration._auth(),
            )
            tweets = response.get("data", [])
            lines = [
                f"Found {response.get('meta', {}).get('result_count', len(tweets))} tweet(s):"
            ]
            for tweet in tweets:
                lines.append(
                    f"  [{tweet.get('id')}] {tweet.get('text', '')[:140]}"
                )
            return "\n".join(lines)

        @mcp.tool()
        def twitter_get_user(username: str) -> str:
            """
            Get a Twitter/X user's profile.

            Args:
                username: Twitter username without @ (e.g. "elonmusk").
            """
            response = integration.get(
                f"{API}/users/by/username/{username}?user.fields=description,public_metrics,verified",
                integration._auth(),
            )
            user = response.get("data", {})
            metrics = user.get("public_metrics", {})
            return integration.ok(
                {
                    "id": user.get("id"),
                    "name": user.get("name"),
                    "username": user.get("username"),
                    "description": user.get("description"),
                    "followers": metrics.get("followers_count"),
                    "following": metrics.get("following_count"),
                    "tweets": metrics.get("tweet_count"),
                }
            )

        @mcp.tool()
        def twitter_get_user_tweets(user_id: str, max_results: int = 10) -> str:
            """
            Get recent tweets from a specific user.

            Args:
                user_id: Twitter user ID (get from twitter_get_user).
                max_results: Number of tweets (5-100, default 10).
            """
            import urllib.parse

            params = urllib.parse.urlencode(
                {
                    "max_results": max(5, min(max_results, 100)),
                    "tweet.fields": "created_at,text",
                }
            )
            response = integration.get(
                f"{API}/users/{user_id}/tweets?{params}",
                integration._auth(),
            )
            tweets = response.get("data", [])
            lines = [f"{len(tweets)} tweet(s) from user {user_id}:"]
            for tweet in tweets:
                lines.append(
                    f"  [{tweet.get('created_at', '')}] {tweet.get('text', '')[:140]}"
                )
            return "\n".join(lines)