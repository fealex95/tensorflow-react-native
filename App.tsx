import { useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import { View, Image, ActivityIndicator } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as tensorFlow from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as FileSystem from 'expo-file-system';
import { decodeJpeg } from '@tensorflow/tfjs-react-native'; 


import {styles} from './styles';

import { Button } from './components/Button';
import { Classification, ClassificationProps } from './components/Classification';

export default function App() {

  const [selectedImageUri, setSelectedImageUri] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<ClassificationProps[]>([]);

  async function handleSelectImage() {
    setIsLoading(true);
    
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true
      })

      if(!result.canceled){
        const {uri} = result.assets[0];
        setSelectedImageUri(uri);
        await imageClassification(uri);
      }
    } catch (error) {
      console.log(error)
    }finally{
      setIsLoading(false);
    }
  }

  async function imageClassification(imageUri: string){
    setResults([]);
    await tensorFlow.ready();
    const model = await mobilenet.load();

    const imageBase64 = await FileSystem.readAsStringAsync(imageUri, {
      encoding: FileSystem.EncodingType.Base64
    })

    const imageBuffer = tensorFlow.util.encodeString(imageBase64, 'base64').buffer;

    const raw = new Uint8Array(imageBuffer);

    const imageTensor = decodeJpeg(raw);

    const classificationResult = await model.classify(imageTensor);

    setResults(classificationResult);
  }

  return (
    <View style={styles.container}>
      <StatusBar style="light"  backgroundColor='transparent' translucent/>

      <Image 
        source={{uri: selectedImageUri ? selectedImageUri : 'https://blog.megajogos.com.br/wp-content/uploads/2018/07/no-image.jpg'}}
        style={styles.image}
      />

      

      <View style={styles.results}>
        {
          results.map((result) => (
            <Classification key={result.className} data={result} />
          ))
        }
          
        
      </View>

      {
        isLoading ?
        <ActivityIndicator color={'#5F1BBF'}/> :
        <Button title='Selecionar imagem' onPress={handleSelectImage} />
      }
      
      
    </View>
  );
}


