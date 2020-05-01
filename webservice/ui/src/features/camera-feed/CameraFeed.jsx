import React from "react";
import "./CameraFeed.css";
import { HTTP, SETTINGS } from "../../constants/constants";

class CameraFeed extends React.Component {
  constructor( props ) {
    super( props );
    this.mjpgSrc = HTTP.CAMERA_FEED;
    this.refreshImage = this.refreshImage.bind( this );
    this.mjpgSrc = HTTP.CAMERA_FEED;

    this.state = {
      mjpgSrc: this.mjpgSrc,
    };
  }
  refreshImage() {
    const d = new Date();
    this.setState( { mjpgSrc: `${ this.mjpgSrc }?ver=${ d.getTime() }` } );
  }

  render() {
    const width = SETTINGS.CAMERA_FEED_WIDTH;//640
    const imgStyle = { "maxWidth": `${ width }px` };
    return (
      <div className="camera-feed" >
        <div className="camera-feed-container">
          <img src={ this.state.mjpgSrc } alt="camera feed" style={ imgStyle } onClick={ this.refreshImage } className="camera-feed-img" />
          <button className="expand-button"><i className="fa fa-arrows-alt fa-2x" aria-hidden="true"></i></button>
        </div>
      </div>
    );
  }
}

export default CameraFeed;
