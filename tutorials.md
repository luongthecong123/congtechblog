---
layout: page
title: Tutorials
permalink: /tutorials/
---

# CUDA Programming Tutorials

Welcome to our comprehensive collection of CUDA programming tutorials. These step-by-step guides will take you from beginner to advanced GPU programming concepts.

## Getting Started

If you're new to CUDA programming, start with these foundational tutorials:

{% assign beginner_tutorials = site.tutorials | where: "difficulty", "Beginner" | sort: "order" %}
{% for tutorial in beginner_tutorials %}
- [{{ tutorial.title }}]({{ tutorial.url | relative_url }}) 
  - **Difficulty**: {{ tutorial.difficulty }}
  - **Duration**: {{ tutorial.duration }}
  - **Prerequisites**: {{ tutorial.prerequisites | join: ", " }}
{% endfor %}

## Intermediate Concepts

Ready to dive deeper? These tutorials cover more advanced topics:

{% assign intermediate_tutorials = site.tutorials | where: "difficulty", "Intermediate" | sort: "order" %}
{% for tutorial in intermediate_tutorials %}
- [{{ tutorial.title }}]({{ tutorial.url | relative_url }})
  - **Difficulty**: {{ tutorial.difficulty }}
  - **Duration**: {{ tutorial.duration }}
  - **Prerequisites**: {{ tutorial.prerequisites | join: ", " }}
{% endfor %}

## Advanced Topics

For experienced CUDA developers looking to optimize performance:

{% assign advanced_tutorials = site.tutorials | where: "difficulty", "Advanced" | sort: "order" %}
{% for tutorial in advanced_tutorials %}
- [{{ tutorial.title }}]({{ tutorial.url | relative_url }})
  - **Difficulty**: {{ tutorial.difficulty }}
  - **Duration**: {{ tutorial.duration }}
  - **Prerequisites**: {{ tutorial.prerequisites | join: ", " }}
{% endfor %}

## Learning Path

We recommend following this learning path:

1. **Foundation** (1-2 weeks)
   - CUDA basics and setup
   - Your first kernel
   - Memory management fundamentals

2. **Core Concepts** (2-3 weeks)
   - Thread hierarchy
   - Shared memory programming
   - Error handling and debugging

3. **Optimization** (3-4 weeks)
   - Performance analysis
   - Memory access patterns
   - Advanced kernel techniques

4. **Specialized Topics** (Ongoing)
   - Multi-GPU programming
   - CUDA libraries
   - Domain-specific applications

## Prerequisites

Before starting these tutorials, ensure you have:

- ✅ NVIDIA GPU with CUDA Compute Capability 3.0+
- ✅ CUDA Toolkit installed
- ✅ C/C++ programming experience
- ✅ Basic understanding of parallel programming concepts

## Need Help?

- 📧 Email us with questions
- 💬 Join our community discussions
- 🐛 Report issues on GitHub
- 📚 Check our FAQ section

---

*New tutorials are added regularly. Subscribe to our [RSS feed]({{ "/feed.xml" | relative_url }}) to stay updated!*
