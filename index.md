---
layout: default
title: Chimera Lab
---

## Recent Articles

{% for post in site.posts %}

<h3 class="article-title">
  <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
</h3>

<div class="article-list-tags">
  {% for tag in post.tags %}
    <span class="article-tag">{{ tag }}</span>
  {% endfor %}
</div>

{% assign word_count = post.content | number_of_words %}
{% assign reading_time = word_count | divided_by: 200.0 | ceil %}

<p class="article-date">
  {{ post.date | date: "%B %-d, %Y" }}
  <span class="meta-separator">|</span>
  <span class="reading-time">{{ reading_time }} min read</span>
</p>

<hr class="article-divider">

{% endfor %}
