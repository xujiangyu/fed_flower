########## path

# nohup sh my_test.sh >my_test_526.log 2>&1 &

### iid
echo "------path---iid----------"
python3.8 xu_test.py --config config/MedPath/DenseNet3DPathIid_test.json

echo "------path---Noniid----------"
python3.8 xu_test.py --config config/MedPath/DenseNet3DPathNoniid_test.json


echo "------path---Unbalance----------"
python3.8 xu_test.py --config config/MedPath/DenseNet3DPathUnbalance_test.json


########## organ

### iid
echo "------organ---iid----------"
python3.8 xu_test.py --config config/MedOrgan/DenseNet3DOrganIid_test.json

echo "------organ---Noniid----------"
python3.8 xu_test.py --config config/MedOrgan/DenseNet3DOrganNoniid_test.json


echo "------organ---Unbalance----------"
python3.8 xu_test.py --config config/MedOrgan/DenseNet3DOrganUnbalance_test.json

########## tissue
echo "------tissue---iid----------"
python3.8 xu_test.py --config config/MedTissue/DenseNetTissueiid_test.json

echo "------tissue---Noniid----------"
python3.8 xu_test.py --config config/MedTissue/DenseNetTissueNoniid_test.json


echo "------tissue---Unbalance----------"
python3.8 xu_test.py --config config/MedTissue/DenseNetTissueUnbalance_test.json

########## vessel
echo "------vessel---iid----------"
python3.8 xu_test.py --config config/MedVessel/DenseNet3DVesseliid_test.json

echo "------vessel---Noniid----------"
python3.8 xu_test.py --config config/MedVessel/DenseNet3DVesselNoniid_test.json


echo "------vessel---Unbalance----------"
python3.8 xu_test.py --config config/MedVessel/DenseNet3DVesselUnbalance_test.json

########## breast
echo "------breast---iid----------"
python3.8 xu_test.py --config config/MedBreast/DenseNetBreastiid_test.json

echo "------breast---Noniid----------"
python3.8 xu_test.py --config config/MedBreast/DenseNetBreastNoniid_test.json

echo "------breast---Noniid1----------"
python3.8 xu_test.py --config config/MedBreast/DenseNetBreastNoniid_test1.json

echo "------breast---Unbalance----------"
python3.8 xu_test.py --config config/MedBreast/DenseNetBreastUnbalance_test.json
