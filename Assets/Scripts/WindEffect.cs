using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WindEffect : MonoBehaviour
{
    public bool inWindZone = false;
    public GameObject windZone;
    float randomStrength;
    public Vector3 randomDirection;
    Quaternion q;
    Rigidbody rb;
    float time;
    float timeDelay;
    // Start is called before the first frame update
    void Start()
    {
        randomStrength = windZone.GetComponent<WindArea>().strength;
        randomDirection = windZone.GetComponent<WindArea>().direction;
        time = 0f;
        
        rb = GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    void Update()
    {
        timeDelay = Random.Range(2.9f, 5.9f);
        time = time + 1f * Time.deltaTime;
        if(time >= timeDelay) {
            time = 0f;
            randomStrength = Random.Range(40.9f, 50.9f);
            randomDirection = new Vector3(Random.insideUnitSphere.x, 0, Random.insideUnitSphere.z);
            if(inWindZone) {
                rb.AddForce(randomStrength * randomDirection, ForceMode.Acceleration);
            }
        }
    }
    
    void FixedUpdate() {
        rb.AddForce(randomDirection * randomStrength);
    }

    // void FixedUpdate() {
    //     if(inWindZone) {
    //         rb.AddForce(windZone.GetComponent<WindArea>().direction * windZone.GetComponent<WindArea>().strength);
    //     }
    // }

    void OnTriggerEnter(Collider coll) {
        if(coll.gameObject.tag == "windArea") {
            windZone = coll.gameObject;
            inWindZone = true;
        }
    }
    void OnTriggerExit(Collider coll) {
        if(coll.gameObject.tag == "windArea") {
            windZone = coll.gameObject;
            inWindZone = false;
        }
    }
}
