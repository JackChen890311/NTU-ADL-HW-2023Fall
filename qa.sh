python run_qa_no_trainer.py \
    --model_name_or_path bert-base-chinese \
    --max_seq_length 512 \
    --doc_stride 128 \
    --output_dir output_qa \
    --train_file data/train.json \
    --validation_file data/valid.json \
    --context_file data/context.json \
    # --dataset_name squad \


# ['id', 'title', 'context', 'question', 'answers']
# To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?
# =====
# Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.
# =====
# {'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]}
# =====