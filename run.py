import tensorflow as tf
import numpy as np

from utils import get_env, arg_parser
from os import environ

from random_agent import RandomAgent

def run(args):
    env = get_env(args.env,
                  results_save_dir=args.results_dir,
                  seed=args.seed)

    sess = tf.Session()
    # TODO: Switch to ACKTR model
    agent = RandomAgent(sess, args, env.action_space.n)
    sess.run(tf.global_variables_initializer())

    global_step = 0
    for ep in xrange(args.num_eps):
        print '-' * 30
        print 'Episode: ', ep
        print '-' * 30

        state = env.reset()

        while True:
            action = agent.get_action(np.expand_dims(state, axis=0))
            state, reward, terminal, _ = env.step(action)

            if args.render:
                env.render()

            if args.train:
                # TODO: Figure out batching.
                global_step = agent.train_step(np.expand_dims(state, axis=0),
                                               np.array([action]),
                                               np.array([reward]),
                                               np.array([terminal]))

            if terminal or global_step > args.num_steps:
                break

        if global_step > args.num_steps:
            break

    # Close the env and write monitor results to disk
    env.close()


if __name__ == '__main__':
    args = arg_parser().parse_args()

    if args.gpu:
        environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    run(args)