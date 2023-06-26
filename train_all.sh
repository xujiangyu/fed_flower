########## path

# nohup sh my_test.sh >my_test_526.log 2>&1 &

### iid
echo "------path---iid----------"
python3.8 xus_server.py --config config/MedPath/DenseNet3DPathIid_0.json &
python3.8 xus_client.py --config config/MedPath/DenseNet3DPathIid_0.json &
python3.8 xus_client.py --config config/MedPath/DenseNet3DPathIid_1.json &
python3.8 xus_client.py --config config/MedPath/DenseNet3DPathIid_2.json &
python3.8 xus_client.py --config config/MedPath/DenseNet3DPathIid_3.json &
python3.8 xus_client.py --config config/MedPath/DenseNet3DPathIid_4.json
sleep 3 

### non-iid
echo "------path---non-iid----------"
python3.8 xus_server.py --config config/MedPath/DenseNet3DPathNoniid_0.json &
python3.8 xus_client.py --config config/MedPath/DenseNet3DPathNoniid_0.json &
python3.8 xus_client.py --config config/MedPath/DenseNet3DPathNoniid_1.json &
python3.8 xus_client.py --config config/MedPath/DenseNet3DPathNoniid_2.json &
python3.8 xus_client.py --config config/MedPath/DenseNet3DPathNoniid_3.json &
python3.8 xus_client.py --config config/MedPath/DenseNet3DPathNoniid_4.json 
sleep 3 

### unbalance
echo "-------path---unbalance----------"
python3.8 xus_server.py --config config/MedPath/DenseNet3DPathUnbalance_0.json &
python3.8 xus_client.py --config config/MedPath/DenseNet3DPathUnbalance_0.json &
python3.8 xus_client.py --config config/MedPath/DenseNet3DPathUnbalance_1.json &
python3.8 xus_client.py --config config/MedPath/DenseNet3DPathUnbalance_2.json &
python3.8 xus_client.py --config config/MedPath/DenseNet3DPathUnbalance_3.json &
python3.8 xus_client.py --config config/MedPath/DenseNet3DPathUnbalance_4.json 
sleep 3 

########## organ

### iid
echo "------organ---iid----------"
python3.8 xus_server.py --config config/MedOrg  an/DenseNet3DOrganIid_0.json &
python3.8 xus_client.py --config config/MedOrgan/DenseNet3DOrganIid_0.json &
python3.8 xus_client.py --config config/MedOrgan/DenseNet3DOrganIid_1.json 
sleep 3 

### non-iid
echo "------organ---non-iid----------"
python3.8 xus_server.py --config config/MedOrgan/DenseNet3DOrganNoniid_0.json &
python3.8 xus_client.py --config config/MedOrgan/DenseNet3DOrganNoniid_0.json &
python3.8 xus_client.py --config config/MedOrgan/DenseNet3DOrganNoniid_1.json
sleep 3 

### unbalance
echo "------organ----unbalance----------"
python3.8 xus_server.py --config config/MedOrgan/DenseNet3DOrganUnbalance_0.json &
python3.8 xus_client.py --config config/MedOrgan/DenseNet3DOrganUnbalance_0.json &
python3.8 xus_client.py --config config/MedOrgan/DenseNet3DOrganUnbalance_1.json 
sleep 3 

########## tissue

### iid
echo "------tissue---iid----------"
python3.8 xus_server.py --config config/MedTissue/DenseNetTissueiid_0.json &
python3.8 xus_client.py --config config/MedTissue/DenseNetTissueiid_0.json &
python3.8 xus_client.py --config config/MedTissue/DenseNetTissueiid_1.json &
python3.8 xus_client.py --config config/MedTissue/DenseNetTissueiid_2.json &
python3.8 xus_client.py --config config/MedTissue/DenseNetTissueiid_3.json &
python3.8 xus_client.py --config config/MedTissue/DenseNetTissueiid_4.json &
python3.8 xus_client.py --config config/MedTissue/DenseNetTissueiid_5.json &
python3.8 xus_client.py --config config/MedTissue/DenseNetTissueiid_6.json &
python3.8 xus_client.py --config config/MedTissue/DenseNetTissueiid_7.json 
sleep 3 

### non-iid
echo "------tissue---non-iid----------"
python3.8 xus_server.py --config config/MedTissue/DenseNetTissueNoniid_0.json &
python3.8 xus_client.py --config config/MedTissue/DenseNetTissueNoniid_0.json &
python3.8 xus_client.py --config config/MedTissue/DenseNetTissueNoniid_1.json &
python3.8 xus_client.py --config config/MedTissue/DenseNetTissueNoniid_2.json &
python3.8 xus_client.py --config config/MedTissue/DenseNetTissueNoniid_3.json &
python3.8 xus_client.py --config config/MedTissue/DenseNetTissueNoniid_4.json &
python3.8 xus_client.py --config config/MedTissue/DenseNetTissueNoniid_5.json &
python3.8 xus_client.py --config config/MedTissue/DenseNetTissueNoniid_6.json &
python3.8 xus_client.py --config config/MedTissue/DenseNetTissueNoniid_7.json 
sleep 3 

### unbalance
echo "------tissue----unbalance----------"
python3.8 xus_server.py --config config/MedTissue/DenseNetTissueUnbalance_0.json &
python3.8 xus_client.py --config config/MedTissue/DenseNetTissueUnbalance_0.json &
python3.8 xus_client.py --config config/MedTissue/DenseNetTissueUnbalance_1.json &
python3.8 xus_client.py --config config/MedTissue/DenseNetTissueUnbalance_2.json &
python3.8 xus_client.py --config config/MedTissue/DenseNetTissueUnbalance_3.json &
python3.8 xus_client.py --config config/MedTissue/DenseNetTissueUnbalance_4.json &
python3.8 xus_client.py --config config/MedTissue/DenseNetTissueUnbalance_5.json &
python3.8 xus_client.py --config config/MedTissue/DenseNetTissueUnbalance_6.json &
python3.8 xus_client.py --config config/MedTissue/DenseNetTissueUnbalance_7.json 
sleep 3 

########## vessel

### iid
echo "-----vessel----iid----------"
python3.8 xus_server.py --config config/MedVessel/DenseNet3DVesseliid_0.json &
python3.8 xus_client.py --config config/MedVessel/DenseNet3DVesseliid_0.json &
python3.8 xus_client.py --config config/MedVessel/DenseNet3DVesseliid_1.json &
python3.8 xus_client.py --config config/MedVessel/DenseNet3DVesseliid_2.json 
sleep 3 

### non-iid
echo "------vessel---non-iid----------"
python3.8 xus_server.py --config config/MedVessel/DenseNet3DVesselNoniid_0.json &
python3.8 xus_client.py --config config/MedVessel/DenseNet3DVesselNoniid_0.json &
python3.8 xus_client.py --config config/MedVessel/DenseNet3DVesselNoniid_1.json &
python3.8 xus_client.py --config config/MedVessel/DenseNet3DVesselNoniid_2.json 
sleep 3 

### unbalance
echo "------vessel----unbalance----------"
python3.8 xus_server.py --config config/MedVessel/DenseNet3DVesselUnbalance_0.json &
python3.8 xus_client.py --config config/MedVessel/DenseNet3DVesselUnbalance_0.json &
python3.8 xus_client.py --config config/MedVessel/DenseNet3DVesselUnbalance_1.json &
python3.8 xus_client.py --config config/MedVessel/DenseNet3DVesselUnbalance_2.json 
sleep 3 

########## breast

### iid
echo "------breast---iid----------"
python3.8 xus_server.py --config config/MedBreast/DenseNetBreastiid_0.json &
python3.8 xus_client.py --config config/MedBreast/DenseNetBreastiid_0.json &
python3.8 xus_client.py --config config/MedBreast/DenseNetBreastiid_1.json 
sleep 3 

### non-iid-1
echo "----breast----non-iid1----------"
python3.8 xus_server.py --config config/MedBreast/DenseNetBreastNoniid_#0.json &
python3.8 xus_client.py --config config/MedBreast/DenseNetBreastNoniid_#1.json &
python3.8 xus_client.py --config config/MedBreast/DenseNetBreastNoniid_#2.json 
sleep 3 

### non-iid-2
echo "------breast--non-iid2----------"
python3.8 xus_server.py --config config/MedBreast/DenseNetBreastNoniid_0.json &
python3.8 xus_client.py --config config/MedBreast/DenseNetBreastNoniid_0.json &
python3.8 xus_client.py --config config/MedBreast/DenseNetBreastNoniid_1.json 
sleep 3 

### unbalance
echo "------breast--unbalance----------"
python3.8 xus_server.py --config config/MedBreast/DenseNetBreastUnbalance_0.json &
python3.8 xus_client.py --config config/MedBreast/DenseNetBreastUnbalance_0.json &
python3.8 xus_client.py --config config/MedBreast/DenseNetBreastUnbalance_1.json 
sleep 3 

########## BrainTumor
### iid
echo "------BrainTumor---iid----------"
python3.8 xus_server.py --config config/Decathon/Unet3DBrainTumorIid_0.json &
python3.8 xus_client.py --config config/Decathon/Unet3DBrainTumorIid_0.json &
python3.8 xus_client.py --config config/Decathon/Unet3DBrainTumorIid_1.json &
python3.8 xus_client.py --config config/Decathon/Unet3DBrainTumorIid_2.json 

python xus_server.py --config config/Decathon/Unet3DBrainTumorIid_0.json &
python xus_client.py --config config/Decathon/Unet3DBrainTumorIid_0.json &
python xus_client.py --config config/Decathon/Unet3DBrainTumorIid_1.json &
python xus_client.py --config config/Decathon/Unet3DBrainTumorIid_2.json 