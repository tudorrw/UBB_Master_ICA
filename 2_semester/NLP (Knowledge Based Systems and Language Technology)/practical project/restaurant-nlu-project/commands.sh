 # Moga Patricia - Ciobanu Sergiu-Tudor

#run rule based extraction 
python -m dstc_da_extraction.extract_data

#run logistic regression to predict DAs
python -m train_predict_DA.train_lr_nlu

python -m train_predict_DA.train_bert_nlu

#generates predicted DAs in a txt file (based on the trained logistic regression model and bert model respectively) to be used for evaluation and for running dialtask
python -m train_predict_DA.generate_predicted -model lr -file predicted.txt 
python -m train_predict_DA.generate_predicted -model bert -file predicted_bert.txt

#evaluate the predicted DAs against the reference DAs using dialtask's evaluation script

python -m dialtask.evaluation.eval_nlu -r data/dstc_extracted_DA/results_nlu_test.json -p train_predict_DA/predicted.txt  

python -m dialtask.evaluation.eval_nlu -r data/dstc_extracted_DA/results_nlu_test.json -p train_predict_DA/predicted_bert.txt  


#run dialtask with the generated predicted DAs to evaluate end-to-end performance

python -m run_dialtask --conf conf/nlu_rules_restaurant.yaml

python -m run_dialtask --conf conf/nlu_dstc_lr.yaml
python -m run_dialtask --conf conf/nlu_dstc_bert.yaml

python -m run_dialtask --conf conf/dstc_rules_restaurant.yaml
python -m run_dialtask --conf conf/dst_dstc_lr.yaml

