import tensorflow as tf

flags = tf.app.flags

# Model Constants
flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
flags.DEFINE_integer("memory_size", 200, "Maximum size of memory.")
flags.DEFINE_integer("epochs", 4000, "Number of epochs to train for.")

# Model Params
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for Adam Optimizer.")
flags.DEFINE_integer("batch_size", 64, "Batch size for training.")
flags.DEFINE_integer("hops", 1, "Number of hops in the Memory Network.")
flags.DEFINE_integer("embedding_size", 32, "Embedding size for embedding matrices.")
flags.DEFINE_integer("soft_weight", 1, "Weight given to softmax function")

# Entity Word Drop
flags.DEFINE_float("word_drop_prob", 0.0, "value to set, if word_drop is set to True")
flags.DEFINE_boolean('word_drop', True, 'if True, drop db words in story')

# PGen Loss
flags.DEFINE_float("p_gen_loss_weight", 1, 'relative weight to p_gen loss, > 1 gives more weight to p_gen loss')
flags.DEFINE_boolean("p_gen_loss", True, 'if True, uses additional p_gen loss during training')

# Model Type
flags.DEFINE_boolean("hierarchy", True, "if True, uses hierarchy pointer attention")
flags.DEFINE_boolean("rnn", True, "if True, uses bi-directional-rnn to encode, else Bag of Words")

# Output and Evaluation Specifications
flags.DEFINE_integer("evaluation_interval", 1, "Evaluate and print results every x epochs")
flags.DEFINE_boolean("bleu_score", True, 'if True, uses BLUE word score to compute best model')
flags.DEFINE_boolean("save", False, "if True, trains using previously saved model")
flags.DEFINE_boolean("debug", False, 'if True, enables debug mode (Verbose Errors, but slower)')
flags.DEFINE_boolean("logging", False, "Turn on Logging for outputs")

# Task Type
flags.DEFINE_integer("task_id", 7, "bAbI task id, 1 <= id <= 8")
flags.DEFINE_boolean('train', False, 'if True, begin to train')
flags.DEFINE_boolean('OOV', False, 'if True, use OOV test set')

# File Locations
flags.DEFINE_string("data_dir", "../data/dialog-bAbI-tasks/", "Directory containing bAbI tasks")
flags.DEFINE_string("logs_dir", "logs/", "Directory containing bAbI tasks")
flags.DEFINE_string("model_dir", "model/", "Directory containing memn2n model checkpoints")
flags.DEFINE_string("kb_file", "../data/dialog-bAbI-tasks/dialog-camrest-kb-all.txt", "kb file for this task")


def get_params():
	return flags.FLAGS

def print_params(logging, args):
	'''
		Print important model parameters
	'''
	print('\n# {}'.format('Model Params'))
	logging.info('[{}] : {}'.format('learning_rate', args.learning_rate))
	logging.info('[{}] : {}'.format('batch_size', args.batch_size))
	logging.info('[{}] : {}'.format('hops', args.hops))
	logging.info('[{}] : {}'.format('embedding_size', args.embedding_size))
	logging.info('[{}] : {}'.format('soft_weight', args.soft_weight))
	
	print('\n# {}'.format('Word Drop'))
	logging.info('[{}] : {}'.format('word_drop', args.word_drop))

	print('\n# {}'.format('PGen Loss'))
	logging.info('[{}] : {}'.format('p_gen_loss', args.p_gen_loss))
	logging.info('[{}] : {}'.format('p_gen_loss_weight', args.p_gen_loss_weight))

	print('\n# {}'.format('Model Type'))
	logging.info('[{}] : {}'.format('hierarchy', args.hierarchy))
	logging.info('[{}] : {}'.format('rnn', args.rnn))

	print('\n# {}'.format('Evaluation Type'))
	logging.info('[{}] : {}'.format('evaluation_interval', args.evaluation_interval))
	logging.info('[{}] : {}'.format('bleu_score', args.bleu_score))
	logging.info('[{}] : {}'.format('save', args.save))
	logging.info('[{}] : {}'.format('debug', args.debug))
	logging.info('[{}] : {}'.format('logging', args.logging))
	
	print('\n# {}'.format('Task Type'))
	logging.info('[{}] : {}'.format('task_id', args.task_id))
	logging.info('[{}] : {}'.format('train', args.train))
	logging.info('[{}] : {}'.format('OOV', args.OOV))

	print('\n# {}'.format('File Locations'))
	logging.info('[{}] : {}'.format('model_dir', args.model_dir))
	logging.info('[{}] : {}'.format('data_dir', args.data_dir.split('/')[-2]))
	logging.info('[{}] : {}'.format('kb_file', args.kb_file))
	


