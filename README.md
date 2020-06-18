# tree_vizzer

Quick visualization of language model attention weights as dependency trees. 

To start, create an environment and install dependencies:
```
conda create -n viz python=3.7
pip install -r requirements.txt
python -m spacy download xx_ent_wiki_sm
```

Example call:
```
python run.py --sentence "The dog chased the cat from the room" --layer 2 --head 3 --attn_dist "mst"
```

This will use the SpaCy multilingual model to tokenize the sentence, encode it with a language model representation (multilingual BERT by default) and serve the dependency tree, using port 5000. The visualization can be accessed by visiting `http://0.0.0.0:5000/` in a browser. Out of the three methods (`mst`, `max`, and `js`), `mst` (naturally) works best in returning valid trees. Both `max` `js` place no tree restriction on the score matrix and can thus produce wonky graphs - often with nodes attaching to themselves. 
