@echo off

echo "000" > run_mnist.log
python mnist.py --model model_000\model.ckpt >> run_mnist.log
echo "001" >> run_mnist.log
python mnist.py --model model_001\model.ckpt >> run_mnist.log
echo "002" >> run_mnist.log
python mnist.py --model model_002\model.ckpt >> run_mnist.log
echo "003" >> run_mnist.log
python mnist.py --model model_003\model.ckpt >> run_mnist.log
echo "004" >> run_mnist.log
python mnist.py --model model_004\model.ckpt >> run_mnist.log
echo "005" >> run_mnist.log
python mnist.py --model model_005\model.ckpt >> run_mnist.log
echo "006" >> run_mnist.log
python mnist.py --model model_006\model.ckpt >> run_mnist.log
echo "007" >> run_mnist.log
python mnist.py --model model_007\model.ckpt >> run_mnist.log
echo "008" >> run_mnist.log
python mnist.py --model model_008\model.ckpt >> run_mnist.log

