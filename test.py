from src.data import *
from src.utils import *
from src.eval_utils import *
from src.embedding import *
from src.metrics import *
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m","--model", 
        default="Fast",
        choices=['Accurate', 'Fast'],
        type=str,
        help="Model to use."
    )
    args = parser.parse_args()

    model_attrs = get_train_model_attributes(model_type=args.model)

    datahandler = DataloaderHandler(
        clip_len=model_attrs.clip_len, 
        alphabet=model_attrs.alphabet, 
        embedding_file=model_attrs.embedding_file,
        embed_len=model_attrs.embed_len
    )

    print("Using trained models to generate outputs for signal prediction training")
    generate_sl_outputs(model_attrs=model_attrs, datahandler=datahandler)

    print("Using trained models to generate outputs of signal prediction")
    y_train, y_train_preds, y_test, y_test_preds = generate_ss_outputs(model_attrs=model_attrs, datahandler=datahandler)
    print(f'''type y_train : {type(y_train)}  , type y_train_preds : {type(y_train_preds)}''')
    y_train_preds.to_csv('y_train_preds.csv')
    y_test.to_csv('y_test.csv')
    y_test_preds.to_csv('y_test_preds.csv')
    y_train.to_csv('y_train.csv')
    print("Generated outputs!")
