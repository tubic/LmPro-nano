# Repo-relative paths (run from the repository root). Set TEXT/DEST to your own
# locations if you keep the tokenised corpus outside the repo.
TEXT=abgnn/nano_data
DEST=abgnn/final_t_data
mkdir -p $DEST

fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/train_t.seq.tokens \
    --validpref $TEXT/valid_t.seq.tokens \
    --testpref $TEXT/test_t.seq.tokens \
    --destdir $DEST/seq \
    --workers 24

fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/train_t.tag.tokens \
    --validpref $TEXT/valid_t.tag.tokens \
    --testpref $TEXT/test_t.tag.tokens \
    --destdir $DEST/tag \
    --workers 24
