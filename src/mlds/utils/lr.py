def get_learning_rate(model_name, task):
    # Decoder		Llama-3.2-1B-Instruct
    # Decoder		Llama-3.2-3B-Instruct
    # Other		NLLB-LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-unsup-simcse
    # Encoder		afriberta_v2_large
    # Encoder-Decoder		afriteva_v2_large
    # Encoder		afro-xlmr-large
    # Encoder		afro-xlmr-large-76L
    # Decoder		gemma-2-2b-it
    # Encoder-Decoder		mt5-large
    # Encoder		xlm-roberta-large
    # Token	Seq

    #     	Dec	        En	        En-Dec	    Other
    # seqc	1.00E-05	1.00E-05	0.00005	    0.0001
    #tokenc	1.00E-05	3.00E-05	0.0001	    0.0003

    encoder = ['afriberta_v2_large', 'afro-xlmr-large', 'afro-xlmr-large-76L', 'xlm-roberta-large']
    decoder = ['Llama-3.2-1B-Instruct', 'Llama-3.2-3B-Instruct', 'gemma-2-2b-it']
    encoder_decoder = ['afriteva_v2_large', 'mt5-large']
    other = ['NLLB-LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-unsup-simcse']

    if any(basename in model_name for basename in encoder):
        return {'tokenc': 3.00E-05, 'seqc': 1.00E-05}[task]
    if any(basename in model_name for basename in decoder):
        return {'tokenc': 1.00E-05, 'seqc': 1.00E-05}[task]
    if any(basename in model_name for basename in encoder_decoder):
        return {'tokenc': 1.00E-04, 'seqc': 5.00E-05}[task]
    if any(basename in model_name for basename in other):
        return {'tokenc': 3.00E-04, 'seqc': 1.00E-04}[task]

    raise ValueError(f'Unknown model name: {model_name}')