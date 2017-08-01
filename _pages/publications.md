---
layout: page
permalink: /publications/
title: Publications
description: Publications and patents by categories in reversed chronological order.
years: [2017, 2016, 2015, 2014, 2013]
---

{% for y in page.years %}
  <h3 class="year">{{y}}</h3>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}
