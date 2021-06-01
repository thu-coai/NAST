# NAST: A Non-Autoregressive Generator with Word Alignment for Unsupervised Text Style Transfer

Work in Progress. 

We will release the code as soon as possible.

## Outputs

We release the outputs of NAST in `outputs`.

```
YELP: outputs/YELP/{stytrans|latentseq}_{simple|learnable}/{pos2neg|neg2pos}.txt
GYAFC: outputs/GYAFC/{stytrans|latentseq}_{simple|learnable}/{fm2inf|inf2fm}.txt
```

* ``{stytrans|latentseq}`` indicates the base model, i.e., [StyTrans](https://arxiv.org/abs/1905.05621) or [LatentSeq](https://arxiv.org/abs/2002.03912).
* ``{simple|learnable}`` indicates the two alignment strategies.


## Citing

Please kindly cite our paper if this paper and the codes are helpful.

```
@inproceedings{huang2021NAST,
  author = {Fei Huang and Zikai Chen and Chen Henry Wu and Qihan Guo and Xiaoyan Zhu and Minlie Huang},
  title = {{NAST}: A Non-Autoregressive Generator with Word Alignment for Unsupervised Text Style Transfer},
  booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics: Findings},
  year = {2021}
}
```