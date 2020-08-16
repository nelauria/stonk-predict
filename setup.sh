mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"choshacao314@gmail.com\"\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml

