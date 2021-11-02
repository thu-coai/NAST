Note: The results below are slightly different from those in our paper because we average 3 trials in the paper.

#### YELP(StyTrans-simple):

```
domain  acc     self_bleu       ref_bleu        ppl   self_g2    self_h2    g2     h2     overall
test0   0.862   0.629   0.491   156.298 0.737   0.727   0.650   0.625   0.650
test1   0.910   0.638   0.633   88.461  0.762   0.750   0.759   0.747   0.759
```

### YELP(StyTrans-learnable):

```
domain  acc     self_bleu       ref_bleu        ppl   self_g2    self_h2    g2     h2     overall
test0   0.860   0.605   0.494   140.370 0.722   0.711   0.652   0.627   0.652
test1   0.902   0.634   0.608   90.799  0.756   0.745   0.740   0.726   0.740
```

#### YELP(LatentSeq-simple):

```
domain  acc     self_bleu       ref_bleu        ppl   self_g2    self_h2    g2     h2     overall
test0   0.816   0.610   0.484   83.284  0.706   0.698   0.628   0.607   0.628
test1   0.810   0.692   0.664   65.338  0.748   0.746   0.733   0.729   0.733
```

#### YELP(LatentSeq-learnable):

```
domain  acc     self_bleu       ref_bleu        ppl   self_g2    self_h2    g2     h2     overall
test0   0.772   0.655   0.530   78.843  0.711   0.709   0.639   0.628   0.639
test1   0.812   0.664   0.646   64.058  0.734   0.731   0.725   0.720   0.725
```

#### GYAFC(StyTrans-simple):

```
domain  acc     self_bleu       ref_bleu        ppl   self_g2    self_h2    g2     h2     overall
test0   0.823   0.688   0.279   161.980 0.752   0.749   0.480   0.417   0.480
test1   0.508   0.627   0.553   104.737 0.564   0.561   0.530   0.529   0.530
```

#### GYAFC(StyTrans-learnable):

```
domain  acc     self_bleu       ref_bleu        ppl   self_g2    self_h2    g2     h2     overall
test0   0.923   0.589   0.313   126.943 0.738   0.720   0.538   0.468   0.538
test1   0.536   0.643   0.556   110.754 0.587   0.585   0.546   0.546   0.546
```

#### GYAFC(LatentSeq-simple):

```
domain  acc     self_bleu       ref_bleu        ppl   self_g2    self_h2    g2     h2     overall
test0   0.581   0.646   0.288   57.074  0.613   0.612   0.409   0.385   0.409
test1   0.587   0.504   0.498   54.621  0.544   0.543   0.540   0.539   0.540
```

#### GYAFC(LatentSeq-learnable):

```
domain  acc     self_bleu       ref_bleu        ppl   self_g2    self_h2    g2     h2     overall
test0   0.630   0.605   0.274   53.733  0.617   0.617   0.416   0.382   0.416
test1   0.580   0.543   0.527   51.850  0.561   0.561   0.553   0.552   0.553
```