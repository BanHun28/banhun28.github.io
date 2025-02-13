FROM ruby

WORKDIR /app

RUN gem install bundler jekyll

COPY entrypoint.sh entrypoint.sh