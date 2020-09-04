from config import parser
global args
args = parser.parse_args()
args_dict = args.__dict__
print("------------------------------------")
print(args.arch+" Configurations:")
for key in args_dict.keys():
    print("- {}: {}".format(key, args_dict[key]))
print("------------------------------------")
