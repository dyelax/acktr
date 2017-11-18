import tensorflow as tf
import numpy as np

from utils import get_env, arg_parser
from os import environ

from random_agent import RandomAgent

def run(args):
    env = get_env(args.env,
                  evaluate=args.evaluate,
                  results_save_dir=args.results_save_dir,
                  seed=args.seed)

    sess = tf.Session()
    # TODO: Switch to ACKTR model
    agent = RandomAgent(sess, env.action_space.n)
    sess.run(tf.global_variables_initializer())

    global_step = 0
    for ep in xrange(args.num_eps):
        state = env.reset()

        # 30 no-ops
        # TODO: This is maybe only for evaluation
        for i in xrange(30):
            state, reward, terminal, _ = env.step(0)

        while True:
            action = agent.get_action(np.array([state]))
            state, reward, terminal, _ = env.step(action)

            if args.render:
                env.render()

            if not args.evaluate:
                # TODO: Figure out batching.
                global_step = agent.train_step(np.array([state]),
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