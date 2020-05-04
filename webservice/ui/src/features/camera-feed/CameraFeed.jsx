import React from "react";
import "./CameraFeed.css";
import { HTTP, SETTINGS } from "../../constants/constants";
import FontAwesome from "react-fontawesome";
import { streamingOn,getPlayerState } from "../../dux/stats";

class CameraFeed extends React.Component {
  constructor(props) {
    super(props);
    
    this.mjpgSrc = HTTP.CAMERA_FEED;
    this.refreshImage = this.refreshImage.bind(this);
    this.mjpgSrc = HTTP.CAMERA_FEED;
    this.isPlaying = getPlayerState();  

    this.state = {
      mjpgSrc: this.mjpgSrc,
      isPlaying:getPlayerState()
    };    
    
    this.togglePlayerState = this.togglePlayerState.bind(this);
    
    document.addEventListener("keypress", function(e) {
      if (e.keyCode === 13) {
        toggleFullScreen();
      }
    }, false);
  }
  refreshImage() {
    const d = new Date();
    this.setState({ mjpgSrc: `${this.mjpgSrc}?ver=${d.getTime()}` });
  }
  togglePlayerState(){
    streamingOn();
    this.isPlaying = getPlayerState();
    console.log(this.isPlaying?'Playing':'Paused');
  }
  toggleFullScreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen();
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen(); 
      }
    }
  }
  render() {
    const width = SETTINGS.CAMERA_FEED_WIDTH;//640
    const height = SETTINGS.CAMERA_FEED_HEIGHT;//640
    const imgStyle = { "maxWidth": `${width}px`, "minHeight": `${height}px` };
    const overlayStyle = { "maxWidth": `${width}px` };
    return (
      <div className="camera-feed" >
        <div className="camera-feed-container">
            <img src={this.state.mjpgSrc} alt="camera feed" style={imgStyle} onClick={this.refreshImage} className="camera-feed-img" />
            <div className="camera-overlay" style={overlayStyle}>
              <div>{this.state.isPlaying} {this.state.mjpgSrc}</div>
              <button className="clearBtn" onClick={this.togglePlayerState}>               
                <FontAwesome name={this.state.isPlaying ? 'play' : 'pause'} size="2x" />
              </button>
              <button className="clearBtn" onClick={this.toggleFullScreen}><FontAwesome name="arrows-alt" size="2x" /></button>
            </div>
        </div>
      </div>
    );
  }
}

export default CameraFeed;
