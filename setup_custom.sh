mkdir -p /home/mlic/mo/baselines/mier_public/data

apt-get update

apt install libosmesa6-dev libgl1-mesa-glx libglfw3 libglu1-mesa libxi6 libgconf-2-4
ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so

mkdir -p ~/.mujoco
cd ~/.mujoco
wget https://www.roboti.us/download/mjpro150_linux.zip
wget https://www.roboti.us/download/mjpro131_linux.zip
wget https://www.roboti.us/file/mjkey.txt
unzip *150*
unzip *131*
rm *zip

if ! grep -q 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin' ~/.bashrc; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin' >> ~/.bashrc
fi
if ! grep -q 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro131/bin' ~/.bashrc; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro131/bin
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro131/bin' >> ~/.bashrc
fi
if ! grep -q 'export PYTHONPATH=/home/mlic/mo/baselines/mier_public' ~/.bashrc; then
    export PYTHONPATH=/home/mlic/mo/baselines/mier_public
    echo 'export PYTHONPATH=/home/mlic/mo/baselines/mier_public' >> ~/.bashrc
fi
if ! grep -q 'export MIER_DATA_PATH=/home/mlic/mo/baselines/mier_public/data' ~/.bashrc; then
    export MIER_DATA_PATH=/home/mlic/mo/baselines/mier_public/data
    echo 'export MIER_DATA_PATH=/home/mlic/mo/baselines/mier_public/data' >> ~/.bashrc
fi

source ~/.bashrc

pip install gym==0.12.0 patchelf mujoco-py==1.50.1.68 pyopengl==3.1.0 \
    tensorflow-probability==0.7.0 gtimer==1.0.0b5 ray==0.8.2 torch==1.0.0 \

echo "Need to update env var. Do \$source ~/.bashrc"