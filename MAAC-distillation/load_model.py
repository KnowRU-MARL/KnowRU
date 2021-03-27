import torch
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id",type=str, default="collect_treasure6-2", help="Name of environment")
    # parser.add_argument("--model_name",
    #                     help="Name of directory to store " +
    #                          "model/training contents")
    parser.add_argument("--n_rollout_threads", default=12, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=50000, type=int)
    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--num_updates", default=4, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for training")
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--use_gpu", action='store_true')

    # base_model
    parser.add_argument("--base_model_dir", type=str, default="base_model/", help="base_model路径")
    # parser.add_argument("--base_model_name", type=list, default=['a_c_0.pt','a_c_0.pt','a_c_1.pt','a_c_2.pt','a_c_1.pt'], help="base_model名字")
    parser.add_argument("--base_model_name", type=list,
                        default=['a_c_0.pt', 'a_c_1.pt', 'a_c_2.pt', 'a_c_3.pt', 'a_c_0.pt', 'a_c_1.pt', 'a_c_4.pt',
                                 'a_c_4.pt'], help="base_model名字")

    parser.add_argument("--alpha", type=float, default="0.5", help="actor_loss的比例")
    config = parser.parse_args()


    base_nets = []
    for i in config.base_model_name:
        str = config.base_model_dir + i
        print(str)
        nets = torch.load(str)
        print(nets)
        base_nets.append(nets)