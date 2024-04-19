$(document).ready(function() {
    const syncedVids = document.querySelectorAll('video.synced');

    let vidsEnded = 0;

    for (const vid of syncedVids) {
        vid.onended = () => {
            vidsEnded += 1;
            console.log(vidsEnded);
            if (vidsEnded === syncedVids.length) {
                setTimeout(() => {
                    vidsEnded = 0;
                    for (const v of syncedVids) {
                        v.currentTime = 0;
                        v.play();
                    }
                }, 1000);
            }
        }
    }
});
