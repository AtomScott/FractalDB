
# Download FractalDB-1k
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1KKqz0H7i_TXFMa2oJtcfry9bmAxyS_SS" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1KKqz0H7i_TXFMa2oJtcfry9bmAxyS_SS" -o FractalDB-1k.tar.gz


# Download FractalDB-60 
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1F0aEogTScpABjJhNZJaCFT-J8mkdP86o" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1F0aEogTScpABjJhNZJaCFT-J8mkdP86o" -o FractalDB-60.tar.gz

# Unzip 
tar -xf FractalDB-1k.tar.gz -C data/
tar -xf FractalDB-60.tar.gz -C data/

# delete downloaded zips
rm FractalDB-1k.tar.gz
rm FractalDB-60.tar.gz