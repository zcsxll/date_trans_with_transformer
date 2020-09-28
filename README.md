# data_trans_with_transformer
a pytorch implementation of machine translation model(transformer) that translates human readable dates ("25th of June, 2009") into machine readable dates ("2009-06-25")

# usage
- Python3 train.py (train with cpu)
- Python3 train.py cuda (train with gpu)
- Python3 test.py (test the model)
 
# attention in decoder(the 2nd multi-head attention)
- I set the number of heads to 2, so there are 2 score maps in the 2nd multi-head attention of decoder
 <p align="left">
<img src="https://github.com/zcsxll/date_trans_with_transformer/blob/master/attention.png" width="600">
</p>
