# CUDA Programming Guide Blog

A comprehensive Jekyll-based technical blog covering CUDA programming, GPU computing, and parallel programming techniques. This site is designed to be hosted on GitHub Pages and provides educational content for developers at all levels.

## 🚀 Quick Start

### Prerequisites
- Ruby 2.7 or higher
- Bundler gem
- Git

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/cuda-programming-guide.git
   cd cuda-programming-guide
   ```

2. **Install dependencies**
   ```bash
   bundle install
   ```

3. **Run the development server**
   ```bash
   bundle exec jekyll serve
   ```

4. **Open your browser**
   Navigate to `http://localhost:4000`

### GitHub Pages Deployment

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Enable GitHub Pages**
   - Go to your repository settings
   - Navigate to "Pages" section
   - Select "Deploy from a branch"
   - Choose "main" branch and "/ (root)" folder
   - Click "Save"

3. **Update site URL**
   Edit `_config.yml` and update the `url` field:
   ```yaml
   url: "https://yourusername.github.io"
   ```

## 📁 Project Structure

```
├── _config.yml          # Jekyll configuration
├── _layouts/             # Page layouts
│   ├── post.html        # Blog post layout
│   └── tutorial.html    # Tutorial layout
├── _posts/              # Blog posts
├── _tutorials/          # Tutorial collection
├── _includes/           # Reusable components
├── assets/             # Static assets (CSS, JS, images)
├── Gemfile             # Ruby dependencies
├── index.md            # Homepage
└── about.md            # About page
```

## ✍️ Creating Content

### Writing Blog Posts

Create a new file in `_posts/` with the format `YYYY-MM-DD-title.md`:

```markdown
---
layout: post
title: "Your Post Title"
date: 2025-01-15
author: "Your Name"
tags: [CUDA, GPU, Programming]
---

Your content here...
```

### Creating Tutorials

Create a new file in `_tutorials/` directory:

```markdown
---
layout: tutorial
title: "Tutorial Title"
date: 2025-01-15
difficulty: Beginner
duration: "30 minutes"
order: 1
prerequisites:
  - "CUDA Toolkit installed"
  - "Basic C/C++ knowledge"
---

Your tutorial content here...
```

## 🎨 Customization

### Site Configuration

Edit `_config.yml` to customize:
- Site title and description
- Author information
- Social links
- Navigation menu

### Styling

The site uses the Minima theme with custom CSS. You can:
- Override theme styles in `assets/main.scss`
- Add custom layouts in `_layouts/`
- Modify includes in `_includes/`

## 📝 Content Guidelines

### Technical Writing
- Use clear, concise explanations
- Include working code examples
- Provide step-by-step instructions
- Add performance considerations

### Code Examples
- Use proper syntax highlighting
- Include error checking in CUDA code
- Provide complete, runnable examples
- Comment code thoroughly

### CUDA Best Practices
- Follow NVIDIA's coding guidelines
- Include memory management examples
- Show both basic and optimized versions
- Discuss performance implications

## 🛠️ Development Tools

### Recommended VS Code Extensions
- Jekyll plugins
- Markdown linting
- YAML support
- Live preview extensions

### Testing Locally
```bash
# Serve with drafts
bundle exec jekyll serve --drafts

# Serve with future posts
bundle exec jekyll serve --future

# Incremental builds
bundle exec jekyll serve --incremental
```

## 🚀 Performance Optimization

### Jekyll Optimization
- Use `incremental: true` for development
- Optimize images before adding
- Minimize custom CSS/JS
- Use Jekyll's built-in compression

### GitHub Pages Optimization
- Keep repository size under 1GB
- Use Jekyll plugins supported by GitHub Pages
- Optimize images and assets
- Enable Jekyll cache

## 📱 Mobile Responsiveness

The site is built with mobile-first design:
- Responsive layouts
- Touch-friendly navigation
- Optimized images
- Fast loading times

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/new-tutorial
   ```
3. **Make your changes**
4. **Test locally**
5. **Submit a pull request**

### Content Contributions
- Follow the writing style guide
- Include proper front matter
- Test all code examples
- Add appropriate tags and categories

## 📊 Analytics and SEO

### Google Analytics
Add your tracking ID to `_config.yml`:
```yaml
google_analytics: UA-XXXXXXXX-X
```

### SEO Optimization
- Use descriptive titles and meta descriptions
- Include relevant keywords in content
- Optimize images with alt text
- Create an XML sitemap (automatically generated)

## 🐛 Troubleshooting

### Common Issues

**Jekyll build fails**
```bash
bundle update
bundle exec jekyll build --verbose
```

**GitHub Pages deployment issues**
- Check GitHub Pages build logs
- Ensure all plugins are GitHub Pages compatible
- Verify `_config.yml` syntax

**Local development problems**
```bash
bundle clean --force
bundle install
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NVIDIA CUDA documentation and examples
- Jekyll and GitHub Pages communities
- Contributors and content reviewers

## 📧 Contact

- **Website**: [Your website URL]
- **Email**: [your.email@example.com]
- **GitHub**: [@yourusername](https://github.com/yourusername)

---

**Happy coding, and may your kernels run fast! 🚀**
