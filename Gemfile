source "https://rubygems.org"

# Use github-pages gem for GitHub Pages compatibility
gem "github-pages", group: :jekyll_plugins

# Windows and JRuby does not include zoneinfo files, so bundle the tzinfo-data gem
# and associated library.
platforms :windows do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

# Lock `http_parser.rb` gem to `v0.6.x` on JRuby builds since newer versions of the gem
# do not have a Java counterpart.
gem "http_parser.rb", "~> 0.6.0", :platforms => [:jruby]
