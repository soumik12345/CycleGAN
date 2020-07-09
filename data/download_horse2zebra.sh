wget https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip
unzip horse2zebra.zip
rm horse2zebra.zip
cd horse2zebra/trainA
wget https://www.dropbox.com/s/hhsdx4ch5wxt6kq/horses_extra.zip
unzip horses_extra.zip
rm horses_extra.zip
mv horse2zebra/trainA/horses_extra_54/* horse2zebra/trainA/
rm -r horse2zebra/trainA/horses_extra_54/